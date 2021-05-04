class Granularity:
    def __init__(self):
        self.sectors = {}
        self.names = {}
        self.sectors["X"] = []
        self.names["X"] = []
        self.sectors["Y"] = []
        self.names["Y"] = []



# tracker used for phase 0 (before 2017)
phaseZeroGranularity = Granularity()        
phaseZeroGranularity.sectors["X"].append( (1,  10)) # BPIX/FPIX
phaseZeroGranularity.sectors["X"].append( (11, 22)) # TIB
phaseZeroGranularity.sectors["X"].append( (23, 34)) # TOB
phaseZeroGranularity.sectors["X"].append( (35, 44)) # TID
phaseZeroGranularity.sectors["X"].append( (45, 64)) # TEC
phaseZeroGranularity.names["X"] = ["PIXEL", "TIB", "TOB", "TID", "TEC"]
phaseZeroGranularity.sectors["Y"].append( (1,  10)) # BPIX/FPIX
phaseZeroGranularity.names["Y"] = ["PIXEL",]

# tracker used for phase 1 (from 2017 until HL-LHC)
# This was a pixel upgrade, so the number pixel sectors changed
phaseOneGranularity = Granularity()        
phaseOneGranularity.sectors["X"].append( (1,  14)) # BPIX/FPIX
phaseOneGranularity.sectors["X"].append( (15, 26)) # TIB
phaseOneGranularity.sectors["X"].append( (27, 38)) # TOB
phaseOneGranularity.sectors["X"].append( (39, 48)) # TID
phaseOneGranularity.sectors["X"].append( (49, 68)) # TEC
phaseOneGranularity.names["X"] = ["PIXEL", "TIB", "TOB", "TID", "TEC"]
phaseOneGranularity.sectors["Y"].append( (1,  14)) # BPIX/FPIX
phaseOneGranularity.names["Y"] = ["PIXEL",]


# this name is used by default by other plotting tools
standardGranularity = phaseOneGranularity
