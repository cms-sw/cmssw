import numpy as np
import ROOT

DIR_BOTH = 0
DIR_UP = 1
DIR_DOWN = -1

NUM_SECTORS = 68
NUM_SECTORS_Y = 14
# systematics has:
        #   a dict with with coordinate names "X", "Y" as keys
        #       - each value of these keys is a list/an array of systematic errors for each sector
        #       - so the list has the length of the number of sectors for that coordinate
        #       - these errors are quadratically added
        #   direction which can be DIR_DOWN, DIR_BOTH, or DIR_UP, depending on whether it adds only down, symmetric or up
        #   isRelative which is a flag telling whether the error is relative to the APE value or absolute    

class SystematicErrors:
    def __init__(self):
        self.X = np.zeros(NUM_SECTORS)
        self.Y = np.zeros(NUM_SECTORS)
        # is not made seperately for X and Y. If this is wanted, make two separate objects
        self.isRelative = np.zeros(NUM_SECTORS, dtype=int) 
        self.direction  = np.empty(NUM_SECTORS, dtype=int) 
        self.direction.fill(DIR_BOTH) # just so it is clear that 
        
    def __getitem__(self, key):
        return getattr(self, key)
    
    def getXFromList(self, X, startat=0):
        for i,x in enumerate(X):
            self.X[i+startat] = x
            
    def getYFromList(self, Y, startat=0):
        for i,y in enumerate(Y):
            self.Y[i+startat] = y
    
    # each line has the structure: xerr yerr isrel direction
    def write(self, fileName):
        with open(fileName, "w") as fi:
            for x, y, rel, direc in zip(self.X, self.X, self.isRelative, self.direction):
                    fi.write("{} {} {} {}".format(x, y, rel, direc))
        
    def read(self, fileName):
        with open(fileName, "r") as fi:
            sector = 0
            for line in fi:
                x, y, rel, direc = line.rstrip().split(" ")
                self.X[sector] = float(x)
                self.Y[sector] = float(y)
                self.isRelative[sector] = int(rel)
                self.direction[sector] = int(direc)
                sector += 1
        return self

# difference between ape values in each sector
# returns a SystematicErrors object with values
def apeDifference(minuend, subtrahend):
    fileA = ROOT.TFile(minuend, "READ")
    fileB = ROOT.TFile(subtrahend, "READ")
    apeTreeA_X = fileA.Get("iterTreeX")
    apeTreeA_X.SetDirectory(0)
    apeTreeB_X = fileB.Get("iterTreeX")
    apeTreeB_X.SetDirectory(0)
    apeTreeA_Y = fileA.Get("iterTreeY")
    apeTreeA_Y.SetDirectory(0)
    apeTreeB_Y = fileB.Get("iterTreeY")
    apeTreeB_Y.SetDirectory(0)
    
    fileA.Close()
    fileB.Close()
    
    # get to last iteration of each tree
    apeTreeA_X.GetEntry(apeTreeA_X.GetEntries()-1)
    apeTreeB_X.GetEntry(apeTreeB_X.GetEntries()-1)
    apeTreeA_Y.GetEntry(apeTreeA_Y.GetEntries()-1)
    apeTreeB_Y.GetEntry(apeTreeB_Y.GetEntries()-1)
    
    difference = SystematicErrors()
    isRel = 0
    direc = 0
    
    for sector in range(1, NUM_SECTORS+1):
        name = "Ape_Sector_{}".format(sector)
        
        diffX = abs(getattr(apeTreeA_X, name) - getattr(apeTreeB_X, name))
        difference.X[sector-1] = diffX
        if sector <= NUM_SECTORS_Y:
            diffY = abs(getattr(apeTreeA_Y, name) - getattr(apeTreeB_Y, name))
            difference.Y[sector-1] = diffY
        difference.isRel[sector-1] = isRel
        difference.direction[sector-1] = direc
    
    return difference


# inFile is allData.root, not allData_iterationApe.root
# returns two arrays with values in x and y
def numberOfHits(inFileName):
    inFile = ROOT.TFile(inFileName, "READ")
    num_x = np.zeros(NUM_SECTORS, dtype=int)
    num_y = np.zeros(NUM_SECTORS, dtype=int)
    for sector in range(1, NUM_SECTORS+1):
        xhist = inFile.Get("ApeEstimator1/Sector_{}/Results/h_ResX".format(sector))
        num_x[sector-1] = xhist.GetEntries()
        if sector <= NUM_SECTORS_Y:
            yhist = inFile.Get("ApeEstimator1/Sector_{}/Results/h_ResY".format(sector))
            num_y[sector-1] = yhist.GetEntries()
    inFile.Close()
    return num_x, num_y

def main():
    pass
    
if __name__ == "__main__":
    main()
