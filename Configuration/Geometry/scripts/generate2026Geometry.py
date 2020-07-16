from Configuration.Geometry.dict2026Geometry import *
from Configuration.Geometry.generateGeometry import *

if __name__ == "__main__":
    # create geometry generator object w/ 2026 content and run it
    generator2026 = GeometryGenerator("generate2026Geometry.py",999,"D","2026",maxSections,allDicts,detectorVersionDict,deprecatedDets,deprecatedSubdets)
    generator2026.run()
