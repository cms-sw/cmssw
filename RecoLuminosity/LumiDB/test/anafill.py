import csv,sys
def main(*args):
   filename=args[1]
   print filename
   csvfile=open(filename,'rb')
   reader=csv.reader(csvfile,delimiter='\t',skipinitialspace=True)
   fill=0
   result=[]
   for row in reader:
      if len(row)!=0:
         fill=row[1]
         result.append(float(row[3]))
   print fill,sum(result)*23.0

if __name__=='__main__':
    sys.exit(main(*sys.argv))
