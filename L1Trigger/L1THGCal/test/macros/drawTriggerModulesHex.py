import ROOT
from DrawingUtilities import Cell, TriggerCell, Module

### PARAMETERS #####
subdet = 4
layer  = 12
zside  = 1
version = 'V8'
inputFileName = "../test_triggergeom.root"
outputFileName = "moduleMap.root"
####################



inputFile = ROOT.TFile.Open(inputFileName)
treeModules      = inputFile.Get("hgcaltriggergeomtester/TreeModules")
treeTriggerCells = inputFile.Get("hgcaltriggergeomtester/TreeTriggerCells")
treeCells        = inputFile.Get("hgcaltriggergeomtester/TreeCells")

## filling cell map
cells = {}
cut = "layer=={0} && zside=={1} && subdet=={2}".format(layer,zside,subdet)
treeCells.Draw(">>elist1", cut, "entrylist")
entryList1 = ROOT.gDirectory.Get("elist1")
entryList1.__class__ = ROOT.TEntryList
nentry = entryList1.GetN()
treeCells.SetEntryList(entryList1)
print '>> Reading cell tree'
for ie in xrange(nentry):
    if ie%10000==0: print " Entry {0}/{1}".format(ie, nentry)
    entry = entryList1.GetEntry(ie)
    treeCells.GetEntry(entry)
    cell = Cell()
    cell.id        = treeCells.id
    cell.zside     = treeCells.zside
    cell.layer     = treeCells.layer
    cell.wafer     = treeCells.wafer
    cell.wafertype = treeCells.wafertype
    cell.cell      = treeCells.cell
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
print '>> Reading trigger cell tree'
for ie in xrange(nentry):
    if ie%10000==0: print " Entry {0}/{1}".format(ie, nentry)
    entry = entryList2.GetEntry(ie)
    treeTriggerCells.GetEntry(entry)
    triggercell = TriggerCell()
    triggercell.id       = treeTriggerCells.id
    triggercell.zside    = treeTriggerCells.zside
    triggercell.layer    = treeTriggerCells.layer
    triggercell.wafer    = treeTriggerCells.wafer
    triggercell.triggercell = treeTriggerCells.triggercell
    triggercell.center.x = treeTriggerCells.x
    triggercell.center.y = treeTriggerCells.y
    triggercell.center.z = treeTriggerCells.z
    for cellid in treeTriggerCells.c_id:
        if not cellid in cells: raise Exception("Cannot find cell {0} in trigger cell".format(cellid))
        cell = cells[cellid]
        triggercell.cells.append(cell)
    triggercells[triggercell.id] = triggercell

## filling module map
modules = {}
treeModules.Draw(">>elist3", cut, "entrylist")
entryList3 = ROOT.gDirectory.Get("elist3")
entryList3.__class__ = ROOT.TEntryList
nentry = entryList3.GetN()
treeModules.SetEntryList(entryList3)
print '>> Reading module tree'
for ie in xrange(nentry):
    if ie%10000==0: print " Entry {0}/{1}".format(ie, nentry)
    entry = entryList3.GetEntry(ie)
    treeModules.GetEntry(entry)
    module = Module()
    module.id       = treeModules.id
    module.zside    = treeModules.zside
    module.layer    = treeModules.layer
    module.module   = treeModules.module
    module.center.x = treeModules.x
    module.center.y = treeModules.y
    module.center.z = treeModules.z
    for cellid in treeModules.tc_id:
        if not cellid in triggercells: raise Exception("Cannot find trigger cell {0} in module".format(cellid))
        cell = triggercells[cellid]
        module.cells.append(cell)
    modules[module.id] = module

## create output canvas
outputFile = ROOT.TFile.Open(outputFileName, "RECREATE")
maxx = -99999.
minx = 99999.
maxy = -99999.
miny = 99999.
for id,cell in cells.items():
    x = cell.center.x
    y = cell.center.y
    if x>maxx: maxx=x
    if x<minx: minx=x
    if y>maxy: maxy=y
    if y<miny: miny=y
minx = minx*0.8 if minx>0 else minx*1.2
miny = miny*0.8 if miny>0 else miny*1.2
maxx = maxx*1.1 if maxx>0 else maxx*0.9
maxy = maxy*1.2 if maxy>0 else maxy*0.8
canvas = ROOT.TCanvas("moduleMap", "moduleMap", 10000, int(10000*(maxy-miny)/(maxx-minx)))
canvas.Range(minx, miny, maxx, maxy)

## Print trigger cells
imod = 0
hexagons = []
print ''
print '>> Drawing subdet', subdet, 'layer', layer, 'zside', zside
print ' Trigger cells are not drawn for all the modules, to reduce the (long) drawing time'
for id,module in modules.items():
    if imod%10==0: print " Drawing module {0}/{1}".format(imod, len(modules))
    imod+=1
    modulelines = []
    for triggercell in module.cells:
        for cell in triggercell.cells:
            hexagon = cell.hexagon()
            cell.hexagon_lines()
            hexagon.SetFillColor(0)
            hexagon.SetLineColor(ROOT.kGray)
            hexagon.Draw()
            hexagons.append(hexagon)
        triggercell.trigger_lines()
        ## only draw trigger cells in some of the modules to save time
        if module.module<100:
            for line in triggercell.lines:
                line.SetLineColor(ROOT.kBlack)
                line.SetLineWidth(2)
                line.Draw()
    module.module_lines()
    for line in module.lines:
        line.SetLineColor(ROOT.kRed)
        line.SetLineWidth(4)
        line.Draw()


canvas.Write()
canvas.Print("moduleMap_{0}_{1}_{2}.png".format(version,subdet,layer))


inputFile.Close()




