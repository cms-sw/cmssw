from Configuration.Geometry.dictCRACKIIGeometry import *
from Configuration.Geometry.generateGeometry import *

if __name__ == "__main__":
    # create geometry generator object w/ Run4 content and run it
    generatorRun4 = GeometryGenerator("generateRun4Geometry.py",999,"D","Run4",maxSections,allDicts,detectorVersionDict)
    generatorRun4.run()
