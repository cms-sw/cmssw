from geometryDiffVisualization import *

g1 = MuonGeometry("geometry1.xml")
g2 = MuonGeometry("geometry2.xml")

draw_station(g1, g2, 3, "tmp.svg", length_factor=100., angle_factor=100.)

draw_wheel(g1, g2, 1, "tmp.svg", length_factor=100., angle_factor=100.)

draw_disk(g1, g2, 1, 3, "tmp.svg", length_factor=100., angle_factor=100.)

