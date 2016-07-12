import ROOT
import random

### PARAMETERS #####
layer  = 27
sector = 1
zside  = 1
inputFileName = "../test_layer26_27.root"
outputFileName = "cellMaps.root"
####################

class Position:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.

class Cell:
    def __init__(self):
        self.id = 0
        self.zside = 0
        self.layer = 0
        self.sector = 0
        self.center = Position()
        self.corners = [Position(), Position(), Position(), Position()]

    def box(self):
        return ROOT.TBox(self.corners[0].x, self.corners[0].y, self.corners[2].x, self.corners[2].y)

    #def box(self):
        #xs = []
        #xs.append(self.corners[0].x)
        #xs.append(self.corners[1].x)
        #xs.append(self.corners[2].x)
        #xs.append(self.corners[3].x)
        #ys = []
        #ys.append(self.corners[0].y)
        #ys.append(self.corners[1].y)
        #ys.append(self.corners[2].y)
        #ys.append(self.corners[3].y)
        #return ROOT.TPolyLine(4, array.array('f',xs), array.array('f',ys))

    def __eq__(self, other):
        return self.id==other.id

class TriggerCell:
    def __init__(self):
        self.id = 0
        self.zside = 0
        self.layer = 0
        self.sector = 0
        self.center = Position()
        self.cells = []


def float_equal(x1, x2):
    prec = 1.e-4
    if abs(x1)<prec and abs(x2)>prec: return False
    elif abs(x1)<prec and abs(x2)<prec: return True
    else: return abs( (x1-x2)/x1)<prec

def compare_lines(line1, line2):
    xy11 = (line1.GetX1(), line1.GetY1())
    xy12 = (line1.GetX2(), line1.GetY2())
    xy21 = (line2.GetX1(), line2.GetY1())
    xy22 = (line2.GetX2(), line2.GetY2())
    samecorner1 = (float_equal(xy11[0],xy21[0]) and float_equal(xy11[1],xy21[1])) or (float_equal(xy11[0],xy22[0]) and float_equal(xy11[1],xy22[1]))
    samecorner2 = (float_equal(xy12[0],xy21[0]) and float_equal(xy12[1],xy21[1])) or (float_equal(xy12[0],xy22[0]) and float_equal(xy12[1],xy22[1]))
    #print "[",xy11,xy12,"]","[",xy21,xy22,"]",(samecorner1 and samecorner2)
    return samecorner1 and samecorner2

def boxlines(box):
    lines = []
    lines.append(ROOT.TLine(box.GetX1(), box.GetY1(), box.GetX1(), box.GetY2()))
    lines.append(ROOT.TLine(box.GetX1(), box.GetY1(), box.GetX2(), box.GetY1()))
    lines.append(ROOT.TLine(box.GetX1(), box.GetY2(), box.GetX2(), box.GetY2()))
    lines.append(ROOT.TLine(box.GetX2(), box.GetY1(), box.GetX2(), box.GetY2()))
    return lines



inputFile = ROOT.TFile.Open(inputFileName)
treeTriggerCells = inputFile.Get("hgcaltriggergeomtester/TreeTriggerCells")
treeCells        = inputFile.Get("hgcaltriggergeomtester/TreeCells")
treeTriggerCells.__class__ = ROOT.TTree
treeCells.__class__ = ROOT.TTree

## filling cell map
cells = {}
cut = "layer=={0} && sector=={1} && zside=={2}".format(layer,sector,zside)
treeCells.Draw(">>elist1", cut, "entrylist")
entryList1 = ROOT.gDirectory.Get("elist1")
entryList1.__class__ = ROOT.TEntryList
nentry = entryList1.GetN()
treeCells.SetEntryList(entryList1)
for ie in xrange(nentry):
    if ie%10000==0: print "Entry {0}/{1}".format(ie, nentry)
    entry = entryList1.GetEntry(ie)
    treeCells.GetEntry(entry)
    cell = Cell()
    cell.id       = treeCells.id
    cell.zside    = treeCells.zside
    cell.layer    = treeCells.layer
    cell.sector   = treeCells.sector
    cell.center.x = treeCells.x
    cell.center.y = treeCells.y
    cell.center.z = treeCells.z
    cell.corners[0].x = treeCells.x1
    cell.corners[0].y = treeCells.y1
    cell.corners[1].x = treeCells.x2
    cell.corners[1].y = treeCells.y2
    cell.corners[2].x = treeCells.x3
    cell.corners[2].y = treeCells.y3
    cell.corners[3].x = treeCells.x4
    cell.corners[3].y = treeCells.y4
    if cell.id not in cells: cells[cell.id] = cell

## filling trigger cell map
triggercells = {}
treeTriggerCells.Draw(">>elist2", cut, "entrylist")
entryList2 = ROOT.gDirectory.Get("elist2")
entryList2.__class__ = ROOT.TEntryList
nentry = entryList2.GetN()
treeTriggerCells.SetEntryList(entryList2)
for ie in xrange(nentry):
    if ie%10000==0: print "Entry {0}/{1}".format(ie, nentry)
    entry = entryList2.GetEntry(ie)
    treeTriggerCells.GetEntry(entry)
    triggercell = TriggerCell()
    triggercell.id       = treeTriggerCells.id
    triggercell.zside    = treeTriggerCells.zside
    triggercell.layer    = treeTriggerCells.layer
    triggercell.sector   = treeTriggerCells.sector
    triggercell.center.x = treeTriggerCells.x
    triggercell.center.y = treeTriggerCells.y
    triggercell.center.z = treeTriggerCells.z
    for cellid in treeTriggerCells.c_id:
        if not cellid in cells: raise StandardError("Cannot find cell {0} in trigger cell".format(cellid))
        cell = cells[cellid]
        triggercell.cells.append(cell)
    triggercells[triggercell.id] = triggercell

print "Read", len(cells), "cells" 
print "Read", len(triggercells), "trigger cells"

## create output canvas
outputFile = ROOT.TFile.Open(outputFileName, "RECREATE")
maxx = -99999.
minx = 99999.
maxy = -99999.
miny = 99999.
for id,triggercell in triggercells.items():
    x = triggercell.center.x
    y = triggercell.center.y
    if x>maxx: maxx=x
    if x<minx: minx=x
    if y>maxy: maxy=y
    if y<miny: miny=y
minx = minx*0.8 if minx>0 else minx*1.2
miny = miny*0.8 if miny>0 else miny*1.2
maxx = maxx*1.2 if maxx>0 else maxx*0.8
maxy = maxy*1.2 if maxy>0 else maxy*0.8
canvas = ROOT.TCanvas("triggerCellMap", "triggerCellMap", 1400, int(1400*(maxy-miny)/(maxx-minx)))
canvas.Range(minx, miny, maxx, maxy)

## Print trigger cells
drawstyle = "lf"
boxes = []
lines = []
for id,triggercell in triggercells.items():
    triggercelllines = []
    for cell in triggercell.cells:
        box = cell.box()
        thisboxlines = boxlines(box)
        for boxline in thisboxlines:
            existingline = None
            for line in triggercelllines:
                if compare_lines(boxline, line):
                    existingline = line
                    break
            if existingline:
                triggercelllines.remove(line)
            else:
                triggercelllines.append(boxline)
        box.SetFillColor(0)
        box.SetLineColor(ROOT.kGray)
        box.Draw(drawstyle)
        boxes.append(box)
        if not "same" in drawstyle: drawstyle += " same"
    lines.extend(triggercelllines)
for line in lines:
    line.Draw()

## Print missing cells
missingboxes = []
for id,cell in cells.items():
    missing = True
    for tid,triggercell in triggercells.items():
        if cell in triggercell.cells:
            missing = False
            break
    if missing:
        box = cell.box()
        box.SetFillColor(0)
        box.SetLineColor(ROOT.kRed-4)
        box.Draw(drawstyle)
        missingboxes.append(box)


canvas.Write()


inputFile.Close()




