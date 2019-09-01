class Granularity:
    def __init__(self):
        self.sectors = {}
        self.names = {}
        self.sectors["X"] = []
        self.names["X"] = []
        self.sectors["Y"] = []
        self.names["Y"] = []

# manually create different granularities here
standardGranularity = Granularity()        
standardGranularity.sectors["X"].append( (1,  14)) # BPIX/FPIX
standardGranularity.sectors["X"].append( (15, 26)) # TIB
standardGranularity.sectors["X"].append( (27, 38)) # TOB
standardGranularity.sectors["X"].append( (39, 48)) # TID
standardGranularity.sectors["X"].append( (49, 68)) # TEC
standardGranularity.names["X"] = ["PIXEL", "TIB", "TOB", "TID", "TEC"]

standardGranularity.sectors["Y"].append( (1,  14)) # BPIX/FPIX
standardGranularity.names["Y"] = ["PIXEL",]
