Changes - timeline
##### 2016.05.11 ATricomi+EMigliore (INFN) changes in PixelChannelIdentifier.cc 
- Change PixelChannelIdentifier::thePacking() to have indexes spanning the full module in case of small pixels (phase II InnerPixel)

##### 2021.08.16. Reserve one bit for the flag in pixel digi packing - see PRs #34509 and #34662
- Change packing from (11, **11**, **0**, 10) -> (11, **10**, **1**, 10), reducing row x column to 2048x1024; and time -> flag

---

#### PixelChannelIdentifier::thePacking( 11, 10, 1, 10); // row, column, flag, adc
// Before phase II changes, it was row=8,col=9, time=4, adc=11  
// Before introducing the flag, it was row=11, col=11, time=0, adc=10

#### Reserved bits:
- row = 11 -> 2^11-1 = 2047
- col = 10 -> 2^10-1 = 1023
- flag = 1 -> 0/1
- adc = 10 -> 2^10-1 = 1023
