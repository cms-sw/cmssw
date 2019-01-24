from __future__ import print_function
etaLUT0=[8.946,7.508,6.279,6.399]
etaLUT1=[-0.159,-0.116,-0.088,-0.128]


#fine eta



for wheel in [-2,-1,0,1,2]:
    for station  in [1,2,3]:
        lut=[]
        for i in range(0,7):
            p=0
            if wheel==0:
                if i==3:
                    p=0
                else:
                    p=i-3
                    p=int(round(etaLUT0[station-1]*p+etaLUT1[station-1]*p*p*p/abs(p)))
            elif (wheel>0):
                p=4+(abs(wheel)-1)*7+6-i
                p=int(round(etaLUT0[station-1]*p+etaLUT1[station-1]*p*p*p/abs(p)))
            else:    
                p=-(4+(abs(wheel)-1)*7+6-i)
                p=int(round(etaLUT0[station-1]*p+etaLUT1[station-1]*p*p*p/abs(p)))
            lut.append(str(p))
        if wheel>0:
            print('etaLUT_plus_{wheel}_{station} = cms.vint32('.format(wheel=wheel,station=station)+','.join(lut)+')\n')
        if wheel<0:
            print('etaLUT_minus_{wheel}_{station} = cms.vint32('.format(wheel=abs(wheel),station=station)+','.join(lut)+')\n')
        if wheel==0:
            print('etaLUT_0_{station} = cms.vint32('.format(station=station)+','.join(lut)+')\n')
 
        #wite HLS LUT    
        HLSINFO={}
        #first the singles
    

        for k in range(0,7):
            tag=pow(2,k) 
            HLSINFO[tag]={'e1':lut[k],'e2':lut[k],'q':1}
        for k in range(0,6):
            for l in range(k+1,7):
                tag=pow(2,k)+pow(2,l)
                HLSINFO[tag]={'e1':lut[k],'e2':lut[l],'q':1}
        keys=sorted(HLSINFO.keys())
        d1=[]
        d2=[]
        d3=[]

        for N in range(0,pow(2,7)):
            if not (N in keys):
                d1.append('0')
                d2.append('0')
                d3.append('0')
            else:
                d1.append(str(HLSINFO[N]['e1']))
                d2.append(str(HLSINFO[N]['e2']))
                d3.append(str(HLSINFO[N]['q']))
                

        if wheel>0:
            etaTag='plus_'+str(abs(wheel))
        elif wheel<0:
            etaTag='minus_'+str(abs(wheel))
        else:
            etaTag='0'
    
        print('const ap_int<8> etaLUT0_'+etaTag+"_"+str(station)+'[128]={'+','.join(d1)+'};\n')
        print('const ap_int<8> etaLUT1_'+etaTag+"_"+str(station)+'[128]={'+','.join(d2)+'};\n')
        print('const ap_int<8> etaLUTQ_'+etaTag+"_"+str(station)+'[128]={'+','.join(d3)+'};\n')
        

            
                
                

#coarse eta
for wheel,p in zip([-2,-1,0,1,2],[-13,-6,0,6,13]):
    lut=[]
    for station  in [1,2,3,4]:
        if p==0:
            lut.append('0')
        else:
            u=int(round(etaLUT0[station-1]*p+etaLUT1[station-1]*p*p*p/abs(p)))
            lut.append(str(u))    


    if wheel>0:
        print('etaCoarseLUT_plus_{wheel}= cms.vint32('.format(wheel=wheel)+','.join(lut)+')\n')
    if wheel<0:
        print('etaCoarseLUT_minus_{wheel} = cms.vint32('.format(wheel=abs(wheel))+','.join(lut)+')\n')
    if wheel==0:
        print('etaCoarseLUT_0 = cms.vint32('+','.join(lut)+')\n')
