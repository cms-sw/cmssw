lut=[]
for k in range(0,8192):
   if k<103:
       fk=103
   else:
       fk=k
   lsb = 1.25 /float(1 << 15);
   fk=fk*lsb
   if fk==0:
       lut.append(str(8191))
   else:
       ptF = 1.0/fk
       pt =int(ptF/0.03125)
       lut.append(str(pt))

print("const ap_uint<BITSSTAPT> ptLUT[8192] = {"+','.join(lut)+'};')

