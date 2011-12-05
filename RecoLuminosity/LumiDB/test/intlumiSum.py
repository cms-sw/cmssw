import csv
def parselumifile(ifilename):
    '''
    input:filename
    output: [(runnumber,delivered)...]
    '''
    result=[]
    try:
        csvfile=open(ifilename,'rb')
        reader=csv.reader(csvfile,delimiter=',',skipinitialspace=True)
        for row in reader:
            runnumber=row[0]
            try:
                runnumber=int(runnumber)
            except ValueError:
                continue
            delivered=float(row[3])
            result.append((runnumber,delivered))
    except Exception,e:
        raise RuntimeError(str(e))
    return result

def lumiuptorun(irunlumi):
    '''
    input: [(runnumber,delivered),...]
    output:[(runnumber,lumisofar),...]
    '''
    intlumiuptorun=[]
    for i,(runnumber,lumival) in enumerate(irunlumi):
        lumivals=[x[1] for x in irunlumi]
        intlumisofar=sum(lumivals[0:i+1])
        intlumiuptorun.append((runnumber,intlumisofar))
    return intlumiuptorun

if __name__=='__main__':
   irunlumimap= parselumifile('/afs/cern.ch/cms/lumi/www/publicplots/totallumivstime-2011.csv')
   intlumitorun=lumiuptorun(irunlumimap)
   print intlumitorun

