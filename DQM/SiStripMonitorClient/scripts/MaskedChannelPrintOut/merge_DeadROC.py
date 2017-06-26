import sys


f1=open('MaskedROC_Run297219.txt') 
f2=open('MaskedROC_Run297211.txt')
f = open('Fill_5949.txt',"w")


name1=[]
name2=[]
namelist = []

for line in f1:
    name1.append(line.strip())

for line in f2:
    name2.append(line.strip())

rocname = list(set(name1) & set(name2))
for i in range(len(rocname)):
    f.write(rocname[i]+"\n")
            
print len(rocname)

f.close()
