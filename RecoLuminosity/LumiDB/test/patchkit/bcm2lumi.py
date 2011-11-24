import csv,sys
def avgval(inarray):
   return sum(inarray)/float(len(inarray))
def writelstofile(filename,inarray):
   f=open(filename,'w')
   for idx,perlsdata in enumerate(inarray):
      f.write(str(idx+1)+' '+str(perlsdata)+' ')
   f.write('\n')
def main(*args):
   filename=args[1]
   outfilename=args[2]
   csvfile=open(filename,'rb')
   reader=csv.reader(csvfile,delimiter=',',skipinitialspace=True)
   result=[]
   perlsdata=[]
   rowcounter=0
   for row in reader:
      if len(row)==0: continue
      rownum=row[0]
      if rownum=='ROW': continue
      calibratedinstlumi=float(row[3])
      calibratedintlumi=calibratedinstlumi*23.357
      if rowcounter!=0 and rowcounter%23==0:
         result.append(avgval(perlsdata))
         perlsdata=[]
      perlsdata.append(calibratedintlumi)
      rowcounter+=1
   print len(result)
   print sum(result)
   writelstofile(outfilename,result)
if __name__=='__main__':
    sys.exit(main(*sys.argv))
