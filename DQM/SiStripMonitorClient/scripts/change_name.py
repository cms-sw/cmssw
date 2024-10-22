#!/usr/bin/env python3


import sys

input_offline=open('DeadROC_offline.txt','r')
input_diff = open('DeadROC_Diff.txt','r')

output = open('MaskedROC_sum.txt','w')


offline =[]
diff = []


for line in input_offline:

   offline.append(str(line.strip())+' power\n')


for line in input_diff:

   diff.append(str(line.strip())+' tbmdelay\n')


full_list = offline + diff

for i in range(len(full_list)):
   output.write(full_list[i])


output.close()
