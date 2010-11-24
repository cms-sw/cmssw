import csv
csvfile=open('spec1509/1509_lumi_CMS.txt','rb')
reader=csv.reader(csvfile,delimiter='\t',skipinitialspace=True)
result=[]
for row in reader:
   if len(row)!=0:
       result.append(float(row[3]))
print sum(result)*23.0

