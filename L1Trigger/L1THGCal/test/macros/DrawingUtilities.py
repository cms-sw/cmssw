import ROOT
import waferGeometry
import math
from array import array

def float_equal(a, b, rel_tol=1e-4, abs_tol=1e-2):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def compare_lines(line1, line2):
    xy11 = (line1.GetX1(), line1.GetY1())
    xy12 = (line1.GetX2(), line1.GetY2())
    xy21 = (line2.GetX1(), line2.GetY1())
    xy22 = (line2.GetX2(), line2.GetY2())
    samecorner1 = (float_equal(xy11[0],xy21[0]) and float_equal(xy11[1],xy21[1])) or (float_equal(xy11[0],xy22[0]) and float_equal(xy11[1],xy22[1]))
    samecorner2 = (float_equal(xy12[0],xy21[0]) and float_equal(xy12[1],xy21[1])) or (float_equal(xy12[0],xy22[0]) and float_equal(xy12[1],xy22[1]))
    #if prt: print "[",xy11,xy12,"]","[",xy21,xy22,"]",(samecorner1 and samecorner2)
    return samecorner1 and samecorner2

def boxlines(box):
    lines = []
    lines.append(ROOT.TLine(box.GetX1(), box.GetY1(), box.GetX1(), box.GetY2()))
    lines.append(ROOT.TLine(box.GetX1(), box.GetY1(), box.GetX2(), box.GetY1()))
    lines.append(ROOT.TLine(box.GetX1(), box.GetY2(), box.GetX2(), box.GetY2()))
    lines.append(ROOT.TLine(box.GetX2(), box.GetY1(), box.GetX2(), box.GetY2()))
    return lines

## remove duplicated lines from a list of lines
def merge_lines(lines):
    copylines = []
    n = len(lines)
    for line1 in lines:
        duplicate = False
        for line2 in lines:
            if line1 is line2: continue
            sameline = compare_lines(line1, line2)
            if sameline: duplicate = True
        if not duplicate:
            copylines.append(line1)
    return copylines

def merge_lines(lines1, lines2):
    rm1 = []
    rm2 = []
    copylines = []
    for i1,line1 in enumerate(lines1):
        sameline = False
        for i2,line2 in enumerate(lines2):
            sameline = compare_lines(line1, line2)
            if sameline: 
                rm2.append(i2)
                break
        if not sameline:
            copylines.append(line1)
    for i,line in enumerate(lines2):
        if i not in rm2:
            copylines.append(line)
    return copylines


class Position:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.

class Cell:
    def __init__(self):
        self.id    = 0
        self.zside = 0
        self.layer = 0
        self.wafer = 0
        self.wafertype = 0
        self.waferrow = 0
        self.wafercolumn = 0
        self.cell = 0
        self.center = Position()
        self.corners = [Position(), Position(), Position(), Position()]
        self.lines = []
        self.edge = 0.

    def point(self):
        return ROOT.TMarker(self.center.x, self.center.y, 1)

    def box(self):
        return ROOT.TBox(self.corners[0].x, self.corners[0].y, self.corners[2].x, self.corners[2].y)

    def hexagon_points(self):
        wafer_geometry = waferGeometry.smallCellWafer if self.wafertype==1 else waferGeometry.largeCellWafer
        self.edge = wafer_geometry['cell_corner_size']
        xs = []
        ys = []
        diameter = self.edge/math.tan(math.radians(30))
        centerx = self.center.x
        centery = self.center.y
        # Shift center for half cells or corner cells
        if self.cell in wafer_geometry['half_cells_edge_left']:
            centerx -=  2.*math.sqrt(3)*self.edge/9.
        elif self.cell in wafer_geometry['half_cells_edge_topleft']:
            centerx -=  2.*math.sqrt(3)*self.edge/9.*math.cos(math.radians(60))
            centery +=  2.*math.sqrt(3)*self.edge/9.*math.sin(math.radians(60))
        elif self.cell in wafer_geometry['half_cells_edge_topright']:
            centerx +=  2.*math.sqrt(3)*self.edge/9.*math.cos(math.radians(60))
            centery +=  2.*math.sqrt(3)*self.edge/9.*math.sin(math.radians(60))
        elif self.cell in wafer_geometry['half_cells_edge_bottomright']:
            centerx +=  2.*math.sqrt(3)*self.edge/9.*math.cos(math.radians(60))
            centery -=  2.*math.sqrt(3)*self.edge/9.*math.sin(math.radians(60))
        elif self.cell in wafer_geometry['half_cells_edge_bottomleft']:
            centerx -=  2.*math.sqrt(3)*self.edge/9.*math.cos(math.radians(60))
            centery -=  2.*math.sqrt(3)*self.edge/9.*math.sin(math.radians(60))
        elif self.cell in wafer_geometry['half_cells_edge_right']:
            centerx +=  2.*math.sqrt(3)*self.edge/9.
        x = centerx - diameter/2.
        y = centery - self.edge/2.
        for angle in range(0, 360, 60):
            y += math.cos(math.radians(angle)) * self.edge
            x += math.sin(math.radians(angle)) * self.edge
            xs.append(x)
            ys.append(y)
        # Remove corners for half cells and corner cells
        if self.cell in wafer_geometry['half_cells_edge_left']:
            del xs[0]; del ys[0]
            del xs[-1]; del ys[-1]
        elif self.cell in wafer_geometry['half_cells_edge_topleft']:
            del xs[0]; del ys[0]
            del xs[0]; del ys[0]
        elif self.cell in wafer_geometry['half_cells_edge_topright']:
            del xs[1]; del ys[1]
            del xs[1]; del ys[1]
        elif self.cell in wafer_geometry['half_cells_edge_bottomright']:
            del xs[3]; del ys[3]
            del xs[3]; del ys[3]
        elif self.cell in wafer_geometry['half_cells_edge_bottomleft']:
            del xs[4]; del ys[4]
            del xs[4]; del ys[4]
        elif self.cell in wafer_geometry['half_cells_edge_right']:
            del xs[2]; del ys[2]
            del xs[2]; del ys[2]
        # Close the cell
        xs.append(xs[0])
        ys.append(ys[0])
        return xs,ys

    def hexagon(self):
        xs,ys = self.hexagon_points()
        return ROOT.TPolyLine(len(xs), array('f',xs), array('f',ys))

    def hexagon_lines(self):
        xs,ys = self.hexagon_points()
        lines = []
        for i in xrange(len(xs)-1):
            lines.append(ROOT.TLine(xs[i], ys[i], xs[i+1], ys[i+1]))
        self.lines = lines
        return lines


    def __eq__(self, other):
        return self.id==other.id


class TriggerCell:
    def __init__(self):
        self.id    = 0
        self.zside = 0
        self.layer = 0
        self.wafer = 0
        self.triggercell = 0
        self.center = Position()
        self.cells = []
        self.lines = []
        self.color = -1

    def trigger_lines(self):
        lines = []
        for cell in self.cells:
            lines = merge_lines(lines, cell.lines)
        self.lines = lines
        return lines


    def __eq__(self, other):
        return self.id==other.id


class Module:
    def __init__(self):
        self.id = 0
        self.zside = 0
        self.layer = 0
        self.module = 0
        self.row = 0
        self.column = 0
        self.center = Position()
        self.cells = []
        self.lines = []

    def module_lines(self):
        lines = []
        for cell in self.cells:
            lines = merge_lines(lines, cell.lines)
        self.lines = lines
        return lines

    def center_from_cells(self):
        x,y,z = 0,0,0
        for cell in self.cells:
            x += cell.center.x
            y += cell.center.y
            z += cell.center.z
        x /= len(self.cells)
        y /= len(self.cells)
        z /= len(self.cells)
        self.center.x = x
        self.center.y = y
        self.center.z = z
        return self.center

