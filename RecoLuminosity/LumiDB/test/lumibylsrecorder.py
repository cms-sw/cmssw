import os,os.path,csv,coral

class lslumiParser(object):
    def __init__(self,lslumifilename,headerfilename):
        '''
        '''
        self.__filename=lslumifilename
        self.__headername=headerfilename
        self.lumidata=[]#[fill,run,lumils,cmsls,beamstatus,beamenergy,delivered,recorded,avgpu]
        self.datatag=''
        self.normtag=''
    def parse(self):
        '''
        parse ls lumi file
        '''
        hf=open(self.__headername,'rb')
        for line in hf:
            if "lumitype" in line:
                fields=line.strip().split(',')
                for field in fields:
                    a=field.strip().split(':')
                    if a[0]=='datatag':
                        self.datatag=a[1].strip()
                    if a[0]=='normtag':
                        self.normtag=a[1].strip()
                break
        hf.close()
        f=open(self.__filename,'rb')
        freader=csv.reader(f,delimiter=',')
        idx=0
        for row in freader:
           if idx==0:
               idx=1 # skip header
               continue
           [run,fill]=map(lambda i:int(i),row[0].split(':'))
           [lumils,cmsls]=map(lambda i:int(i),row[1].split(':'))
           chartime=row[2]
           beamstatus=row[3]
           beamenergy=float(row[4])
           delivered=float(row[5])
           recorded=float(row[6])
           avgpu=float(row[7])
           self.lumidata.append([fill,run,lumils,cmsls,chartime,beamstatus,beamenergy,delivered,recorded,avgpu])
        f.close()
    
if __name__ == "__main__" :
    lslumifilename='209089_data.csv'
    lumiheaderfilename='209089_header.txt'
    p=lslumiParser(lslumifilename,lumiheaderfilename)
    p.parse()
    print p.datatag,p.normtag
    print p.lumidata
