#!/usr/bin/env python
import commands,string,time,thread,random,math,sys

#global variables
MINRES=0.02
SEED=-1
ENDCAPSINGLE=0
NPOINT=3
eta=[]
eta_ev=[]
res=[]
#
eta.append((0.+0.261)/2)
eta.append((0.957+0.783)/2)
eta.append((1.479+1.305)/2)

# events/crystal/fb


def miscalib(lumi,endcap,z,etaindex,phiindex,randval):
    global MINRES,ENDCAPSINGLE,NPOINT
    if endcap:
        modindex=etaindex
        crysindex=phiindex
    #return a random factor according to  eta
    if lumi==0:
        if randval==1:
            return random.gauss(1,MINRES)
#            return 1.0
        else:
            return MINRES
    if endcap != 1:
        # from AN 
        #0.000 0.261         6.19 0.12
        #0.783 0.957        10.7 0.27
        #1.305 1.479        15.0 0.42
        # get eta from etaindex
        etacur=(etaindex-0.5)*0.0174
        eta_ev.append(300./5*lumi)   
        eta_ev.append(270./5*lumi)
        eta_ev.append(170./5*lumi) 
        res.append(math.sqrt(6.19**2/eta_ev[0]+0.12**2))
        res.append(math.sqrt(10.7**2/eta_ev[1]+0.27**2))
        res.append(math.sqrt(15**2/eta_ev[2]+0.42**2))
        # extrapolation
        
        for ieta in range(0,NPOINT):
            if eta[ieta]> etacur:
                break
        indexmin=ieta-1
        indexmax=ieta
        # in case there are no more points left
        if(indexmin<0):
            indexmin=indexmin+1
            indexmax=indexmax+1
        # now real extrapolation
        real_res=res[indexmin]+(etacur-eta[indexmin])*(res[indexmax]-res[indexmin])/(eta[indexmax]-eta[indexmin])
        real_res=real_res/100
        if real_res>MINRES:
            real_res=MINRES
        if randval==1:
            return random.gauss(1,real_res)
        else:
            return real_res
    if endcap:
        if ENDCAPSINGLE==0:
            # first time called, we have to derive it from last barrel
            ENDCAPSINGLE=miscalib(lumi,0,1,85,1,0)
        if randval==1:
            return random.gauss(1,ENDCAPSINGLE)
        else:
            return ENDCAPSINGLE

    

#main
if len(sys.argv)==1:
    print 'Usage: '+sys.argv[0]+' <barrel|endcap> <lumi> <filename> [MINRES=0.02] [SEED=random]'
    print '       put lumi=0 for precalibration values (random at MINRES)'
    sys.exit(1)

if sys.argv[1]=='barrel':
    endcap=0
elif sys.argv[1]=='endcap':
    endcap=1
else:
    print 'please specify one of <barrel|endcap>'
    sys.exit(1)

if endcap==1:
    # read non existing cells file
    ne_z=[]
    ne_mod=[]
    ne_xtal=[]
    necells=0

#    necell=open('non_existing_cell_endcap','r')

#    for line in necell:
#        necells=necells+1
#        curr_list=line.split()
#        ne_z.append(string.atoi(curr_list[0]))
#        ne_mod.append(string.atoi(curr_list[1]))
#        ne_xtal.append(string.atoi(curr_list[2]))
#    necell.close()
#    print 'Read ',necells,' non-existing cells for endcap'

    
lumi=string.atof(sys.argv[2])

if lumi<0:
    print 'lumi = '+str(lumi)+' not valid'
    sys.exit(1)
    
fileout=sys.argv[3]

if len(sys.argv)>=5:
    MINRES=string.atof(sys.argv[4])

print 'Using minimal resolution: '+str(MINRES)

if len(sys.argv)>=6:
    SEED=string.atoi(sys.argv[5])
    print 'Using fixed seed for random generation: '+str(SEED)
    random.seed(SEED)
    
# now open file
xmlfile=open(fileout,'w')
# write header
xmlfile.write('<?xml version="1.0" ?>\n')
xmlfile.write('\n')
xmlfile.write('<CalibrationConstants>\n')
if endcap==0:
    xmlfile.write('  <EcalBarrel>\n')
else:
    xmlfile.write('  <EcalEndcap>\n')
# define ranges
mineta=1
minphi=1
if endcap==0:
    maxeta=85
    maxphi=360
else:
    maxeta=100
    maxphi=100

count=0
for zindex in (-1,1):
    for etaindex in range(mineta,maxeta+1):
        for phiindex in range(minphi,maxphi+1):
            miscal_fac=miscalib(lumi,endcap,zindex,etaindex,phiindex,1)
            # create line:
            if endcap==0:
	       if zindex==-1:
                  line='        <Cell eta_index="'+str(-etaindex)+'" phi_index="'+str(phiindex)+'" scale_factor="'+str(miscal_fac)+'"/>\n'
                  xmlfile.write(line)
                  count=count+1
	       else:
                  line='        <Cell eta_index="'+str(+etaindex)+'" phi_index="'+str(phiindex)+'" scale_factor="'+str(miscal_fac)+'"/>\n'
                  xmlfile.write(line)
                  count=count+1
            else:
                goodxtal=1
                # check if it exists
#                for bad in range(0,necells):
#                    if ne_xtal[bad]==phiindex:
#                        if ne_mod[bad]==etaindex:
#                            if ne_z[bad]==zindex:
#                                goodxtal=0
#                                break
#                if goodxtal==1:
#		   if zindex==-1:

#                line='        <Cell module_index="'+str(etaindex)+'" crystal_index="'+str(phiindex)+'"  scale_factor="'+str(miscal_fac)+'"/>\n'
                line='        <Cell x_index="'+str(etaindex)+'" y_index="'+str(phiindex)+'" z_index="'+str(zindex)+'" scale_factor="'+str(miscal_fac)+'"/>\n'

                xmlfile.write(line)
                count=count+1
                #                   else :
                #                      line='        <Cell module_index="'+str(etaindex)+'" crystal_index="'+str(phiindex)+'"  scale_factor="'+str(miscal_fac)+'"/>\n'
                #		      xmlfile.write(line)
                #                      count=count+1

# write footer
if endcap==0:
    xmlfile.write('  </EcalBarrel>\n')
else:
    xmlfile.write('  </EcalEndcap>\n')
xmlfile.write('</CalibrationConstants>\n')
xmlfile.close()
print 'File '+fileout+' written with '+str(count)+' lines'
#print miscalib(5,0,1,85,1,0)
#print miscalib(5,1,-1,10,1,0)
#print miscalib(5,1,-1,10,1,1)
#print miscalib(5,1,-1,10,1,1)
