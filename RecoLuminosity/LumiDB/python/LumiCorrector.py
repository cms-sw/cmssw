##
##identical copy from online. no drift correction
##
class LumiCorrector(object):
    def __init__(self,norm=7.13,punorm=6.37,alpha1=0.063,alpha2=-0.0037):
        self.Norm_=norm
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
    def AfterglowFactor(self,nBXs):
        Afterglow = 1.0
        for bxthreshold,correction in self.AfterglowMap_.items():
            if nBXs >= bxthreshold :
                Afterglow = correction
                return Afterglow
        return Afterglow
    def TotalCorrectionFactor(self,TotLumi_noNorm,nBXs):
        if nBXs==0: return 1.0
        AvgLumi = self.PUNorm_*TotLumi_noNorm/nBXs
        return self.Norm_*self.AfterglowFactor(nBXs)/(1 + self.Alpha1_*AvgLumi + self.Alpha2_*AvgLumi*AvgLumi)
    
if __name__=='__main__':
    lcorr=LumiCorrector()
    print lcorr.AfterglowFactor(500)
    print lcorr.TotalCorrectionFactor(0.3,700)
