##
##identical copy from online. no drift correction
##
class LumiCorrector(object):
    def __init__(self,occ1norm=7.13e3,occ2norm=7.97e3,etnorm=1.59e3,punorm=6.37e3,alpha1=0.063,alpha2=-0.0037):
        self.Occ1Norm_=occ1norm
        self.Occ2Norm_=occ2norm
        self.ETNorm_=etnorm
        self.PUNorm_=punorm
        self.Alpha1_=alpha1
        self.Alpha2_=alpha2
        self.AfterglowMap_={}
        self.AfterglowMap_[213]=0.992 
        self.AfterglowMap_[321]=0.990 
        self.AfterglowMap_[423]=0.988 
        self.AfterglowMap_[597]=0.985 
        self.AfterglowMap_[700]=0.984 
        self.AfterglowMap_[873]=0.981 
        self.AfterglowMap_[1041]=0.979 
        self.AfterglowMap_[1179]=0.977 
        self.AfterglowMap_[1317]=0.975
        self.pixelAfterglowMap_={}
        self.pixelAfterglowMap_[213]=0.989 
        self.pixelAfterglowMap_[321]=0.989 
        self.pixelAfterglowMap_[423]=0.985 
        self.pixelAfterglowMap_[597]=0.983 
        self.pixelAfterglowMap_[700]=0.980 
        self.pixelAfterglowMap_[873]=0.980 
        self.pixelAfterglowMap_[1041]=0.976 
        self.pixelAfterglowMap_[1179]=0.974 
        self.pixelAfterglowMap_[1317]=0.972
    def setNormForAlgo(self,algo,value):
        if algo=='OCC1':
            self.Occ1Norm_=value
            return
        if algo=='OCC2':
            self.Occ2Norm_=value
            return
        if algo=='ET':
            selfETNorm_=value
            return
        if algo=='PU':
            self.PUNorm_=value
            return
    def setCoefficient(self,name,value):
        if name=="ALPHA1":
            self.Alpha1_=value
        if name=="ALPHA2":
            self.Alpha2_=value
    def getNormForAlgo(self,algo):
        if algo=="OCC1" :
            return self.Occ1Norm_
        if algo=="OCC2" :
            return self.Occ2Norm_
        if algo=="ET" :
            return self.ETNorm_
        if algo=="PU" :
            return self.PUNorm_
        return 1.0
    def getCoefficient(self,name):
        if name=="ALPHA1" :
            return Alpha1_
        if name=="ALPHA2":
            return Alpha2_
        return 0.0
    
    def AfterglowFactor(self,nBXs):
        Afterglow = 1.0
        for bxthreshold,correction in self.AfterglowMap_.items():
            if nBXs >= bxthreshold :
                Afterglow = correction
                return Afterglow
        return Afterglow
    
    def TotalNormOcc1(self,TotLumi_noNorm,nBXs):
        AvgLumi = 0.
        if nBXs>0:
            AvgLumi = self.PUNorm_*TotLumi_noNorm/nBXs            
        else:
            return 1.0
        return self.Occ1Norm_*self.AfterglowFactor(nBXs)/(1 + self.Alpha1_*AvgLumi + self.Alpha2_*AvgLumi*AvgLumi)
    
    def PixelAfterglowFactor(self,nBXs):
        Afterglow = 1.0
        for bxthreshold,correction in self.pixelAfterglowMap_.items():
            if nBXs >= bxthreshold :
                Afterglow = correction
                return Afterglow
        return Afterglow
if __name__=='__main__':
    lcorr=LumiCorrector()
    print lcorr.AfterglowFactor(500)
    print lcorr.TotalNormOcc1(0.3,700)
    print lcorr.PixelAfterglowFactor(500)
