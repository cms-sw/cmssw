import os, sys, re
import string
import math
from ROOT import *

usage = "usage: %s python EvalPdfUnc.py infile outfile" %         os.path.basename(sys.argv[0])

if len(sys.argv) < 3:
   print usage
   sys.exit(2)
else:
   argv = sys.argv
   print argv
   infile = argv[1]
   outfile = argv[2]
   print argv[1]
   f = open(infile, 'read')
   ## outfile in append mode
   ff = open(outfile,'a')



acc = std.vector(float)()
acc_rew = std.vector(float)()
nevt = std.vector(float)()
nevt_rew = std.vector(float)()
line= f.readline()
while line:
    l=line.split()
    acc.push_back( float(l[0]))
    acc_rew.push_back( float(l[1]) )
    nevt.push_back( float(l[2]))
    nevt_rew.push_back( float(l[3]))
    line= f.readline()
f.close()
#for i in acc_rew:
#    print i


def mean ( v):
  mean = 0.
  for e in v:
      mean+= e
  n = float(v.size())    
  return ( mean / n )   

avg_acc =   mean (acc)
print "mean acc: ", avg_acc
avg_acc_rew =   mean (acc_rew)
print "mean acc_rew: ", avg_acc_rew
diff_acc= (avg_acc_rew - avg_acc) / ( avg_acc )
avg_nevt =   mean (nevt)
print "mean nevt: ", avg_nevt
avg_nevt_rew =   mean (nevt_rew)
print "mean nevt_rew: ", avg_nevt_rew
diff_evt =  (avg_nevt_rew - avg_nevt) / ( avg_nevt ) 


def eval_asym_sys(eff):
  ## asym error according to Hof's master formula  
   d1 = 0.
   d2 = 0.
   x0 = 0.
   s1 = 0.
   s2 = 0.
   for idx in range(len(eff)) :
    i = eff[idx]
    if idx == 0 :
        x0 = i
    else :
        if idx % 2 != 0:
            d1 =  i - x0
        else :
            d2 =  x0 - i  
            if(d1 < 0) :
                #if (d2<0):
                 #   d1=0
                 #   d2=0
                tmp = d1
                d1 = -d2
                d2 = -tmp
            print idx/2, ' ' , x0, '[+', d1, ' -', d2, ']'    
        m1 = max(d1, 0.)   
        s1 += m1*m1
        m2 = max(d2, 0.)   
        s2 += m2*m2
   s1 = sqrt(s1)
   s2 = sqrt(s2)
   print >> ff, infile, 'asym error'
   print >> ff ,   ' x = ', x0, '[+', s1, ' -', s2, ']'
   print >> ff,  'err = +', s1/x0*100, ' -', s2/x0*100
  


def eval_max_asym_sys(eff):
   ## symmetrizing the error, taking each step the max between the two asym errors  
   d1 = 0.
   d2 = 0.
   x0 = 0.
   s1 = 0.
   for idx in range(len(eff)) :
    i = eff[idx]
    if idx == 0 :
        x0 = i
    else :
        if idx % 2 != 0:
            d1 =  i - x0
        else :
            d2 =  x0 - i  
            if(d1 < 0) :
                #if (d2<0):
                 #   d1=0
                 #   d2=0
                tmp = d1
                d1 = -d2
                d2 = -tmp
            print idx/2, ' ' , x0, '[+', d1, ' -', d2, ']'    
        m = max(abs(d1), abs(d2))   
        s1 += m*m
   s1 = sqrt(s1)
   print >> ff, infile, 'sym error taking the max between asym errors'
   print >> ff ,   ' x = ', x0, '[+', s1, ' -', s1, ']'
   print >> ff,  'err = +', s1/x0*100, ' -', s1/x0*100
  



def eval_mean_asym_sys(eff):
   ## symmetrizing the error, taking each step the mean between the two asym errors  
   d1 = 0.
   d2 = 0.
   x0 = 0.
   s1 = 0.
   for idx in range(len(eff)) :
    i = eff[idx]
    if idx == 0 :
        x0 = i
    else :
        if idx % 2 != 0:
            d1 =  i - x0
        else :
            d2 =  x0 - i  
            if(d1 < 0) :
                #if (d2<0):
                 #   d1=0
                 #   d2=0
                tmp = d1
                d1 = -d2
                d2 = -tmp
            print idx/2, ' ' , x0, '[+', d1, ' -', d2, ']'    
        m = 0.5 * ( abs(d1) + abs(d2))   
        s1 += m*m
   s1 = sqrt(s1)
   print >> ff, infile, 'sym error taking the mean between asym errors'
   print >> ff ,   ' x = ', x0, '[+', s1, ' -', s1, ']'
   print >> ff,  'err = +', s1/x0*100, ' -', s1/x0*100
  


def eval_quadsum_asym_sys(eff):
   ## symmetrizing the error, taking each step the quadractic sum between the two asym errors  
   d1 = 0.
   d2 = 0.
   x0 = 0.
   s1 = 0.
   for idx in range(len(eff)) :
    i = eff[idx]
    if idx == 0 :
        x0 = i
    else :
        if idx % 2 != 0:
            d1 =  i - x0
        else :
            d2 =  x0 - i  
            if(d1 < 0) :
                #if (d2<0):
                 #   d1=0
                 #   d2=0
                tmp = d1
                d1 = -d2
                d2 = -tmp
            print idx/2, ' ' , x0, '[+', d1, ' -', d2, ']'    
        m = sqrt( 0.5 *( abs(d1)* abs(d1) + abs(d2)*abs(d2)) )   
        s1 += m*m
   s1 = sqrt(s1)
   print >> ff, infile, 'sym error taking the qaudratic sum between asym errors'
   print >> ff ,   ' x = ', x0, '[+', s1, ' -', s1, ']'
   print >> ff,  'err = +', s1/x0*100, ' -', s1/x0*100


sys_acc_asym = eval_asym_sys(acc_rew)
sys_nevt_asym = eval_asym_sys(nevt_rew )


sys_acc_max_asym = eval_max_asym_sys(acc_rew)
sys_nevt_max_asym = eval_max_asym_sys(nevt_rew )

sys_acc_mean_asym = eval_mean_asym_sys(acc_rew)
sys_nevt_mean_asym = eval_mean_asym_sys(nevt_rew  )

sys_acc_quadsum_asym = eval_quadsum_asym_sys(acc_rew)
sys_nevt_quadsum_asym = eval_quadsum_asym_sys(nevt_rew )





#print "sys acc:",  acc_rew[0], " +- " , sys_acc
#print "sys nevt:",  nevt_rew[0], " +- " , sys_nevt

