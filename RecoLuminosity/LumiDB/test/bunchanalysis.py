import sys,os,os.path,csv
#analyse result from lumicalc2 cmmd to find perbunchlumi after subtracting noise
#the following example analyses the output from:
#lumiCalc2.py -r 193092 lumibylsXing -o 193092-bx.csv --without-correction --xingMinLum 0
#
class bxlumiParser(object):
    def __init__(self,bxfilename,cmsonly=False):
        '''
        if cmsonly, we do not consider lumi sections outside cms run
        '''
        self.__filename=bxfilename
        self.__cmsonly=cmsonly#if cmsonly=True, consider only lumi section where cms daq is on
        self.bxlumi={}#{bxidx:lumi}
        self.bxlumimax=0.
        self.lssum={}#not used, just in case per ls info is needed
        self.lsmax={}#not used, just in case per ls info is needed
    def parse(self):
        '''
        parse perbunch lumi file
        '''
        f=open(self.__filename,'rb')
        freader=csv.reader(f,delimiter=',')
        idx=0
        for row in freader:
            if idx==0:
                idx=1 #skip header
                continue
            [run,fill]=map(lambda i:int(i),row[0].split(':'))
            [lumils,cmsls]=map(lambda i: int(i),row[1].split(':'))
            
            
            if self.__cmsonly and int(cmsls)==0: continue
            if len(row)<6 : #
                self.lssum[lumils]=0.
                self.lsmax[lumils]=0.
                continue
            bxdata=row[5:]
            bxidx=[int(bxdata[i]) for i in range(len(bxdata)) if i%2 ==0]
            bxlum=[float(bxdata[i]) for i in range(len(bxdata)) if i%2 !=0]
            self.lssum[lumils]=sum(bxlum)
            self.lsmax[lumils]=max(bxlum)
            for i,inxval in enumerate(bxidx):
                if not self.bxlumi.has_key(inxval): self.bxlumi[inxval]=0.
                self.bxlumi[inxval]+=bxlum[i]
        f.close()
        if self.bxlumi:
            self.bxlumimax=max(self.bxlumi.values())
            
    def iscollidingbx(self,bxidx):
        '''
           collidingbx is the bx with lumi within 20% of the max lumi
           noncollidingbx is the bunch with lumi >20% off the peak
        '''
        if not self.bxlumi.has_key(bxidx): return False
        if self.bxlumi[bxidx] < self.bxlumimax*0.2: return False
        return True
    
if __name__ == "__main__" :
    ifilename='193092-bx.csv'
    bxreader=bxlumiParser(ifilename,cmsonly=False)
    bxreader.parse()
    collidingbx=[i for i in range(0,3564) if bxreader.iscollidingbx(i)]
    noncollidingbx=[i for i in range(0,3564) if i not in collidingbx]
    print 'colliding bx ',collidingbx
    noise=0.
    print 'bxidx,rawbxlumi,truebxlumi'
    totlumi=0.
    totncolliding=0
    for i in collidingbx:
        rawbxlumi=bxreader.bxlumi[i]
        previousnoncollidingbx=[j for j in noncollidingbx if j<i]
        if previousnoncollidingbx:
            noise=bxreader.bxlumi[max(previousnoncollidingbx)]
        truebxlumi=rawbxlumi-noise
        totncolliding+=1
        totlumi+=truebxlumi
        print i,rawbxlumi,truebxlumi
    print '==n clean colliding bx, lumi ',totncolliding,totlumi
