#!/bin/bash

# Masks some undefined bits in data dump from EcalDumpRaw or EcalDataReader.
# Data interpretation is also dropped. This script can be used to prepare 
# two data dumps which are wanted to be compared with the diff utility.
# Original author: Ph. Gras CEA/Saclay

awk '
/^\[00000000\]/{fed=$15; print fed;}
/^\[/ {
add=strtonum("0x" substr($1, 2, 8));
a=strtonum("0x" $2); 
b=strtonum("0x" $3); 
c=strtonum("0x" $4); 
d=strtonum("0x" $5);
if((fed < 610) || (fed > 645)){
  if(add==0x48 || add==0x90 || add==0xD8 || add==0x120) c = and(c,0xFFFF - lshift(1,12));
  if(add==0x88 || add==0xa8 || add==0xb0 || add==0xc8 || add==0xd0 || add==0x118 || add==0x138 || add==0x140 || add==0x158 || add==0x160) a = b = c = d = 0;
  if(add==0x190) c = 0;
} else{
  if(add==0x48)  c = and(c, 0xFFFF - lshift(1,12));
  if(add==0x100) c = 0;
}
if($0 ~ /TTS/) c = 0;
printf "[%08x] %04x %04x %04x %04x\n", add,a,b,c,d;
}'