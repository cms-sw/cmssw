2016.05.11 ATricomi+EMigliore (INFN) changes in PixelChannelIdentifier.cc 
Change PixelChannelIdentifier::thePacking() to have indexes spanning the full module in case of small pixels (phase II InnerPixel)

PixelChannelIdentifier::thePacking( 11, 11, 0, 10); // It was row=8,col=9, time=4, adc=11

Reserved bits
row = 11 -> 2^11-1 = 2047
col = 11 -> 2^11-1 = 2047
time = 0 (not used)
adc = 10 -> 2^10-1 = 1023

