from Configuration.Geometry.dict2021Geometry import *
from Configuration.Geometry.generateGeometry import *

if __name__ == "__main__":
    # create geometry generator object w/ 2021 content and run it
    generator2021 = GeometryGenerator("generate2021Geometry.py",2021,"","",maxSections,allDicts,detectorVersionDict,detectorVersionType=str,deprecatedSubdets=deprecatedSubdets)
    generator2021.run()


