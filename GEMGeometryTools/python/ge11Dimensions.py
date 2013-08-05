# configuration file to produce GEM XML geometries

## number of eta partitions
dzEta = [7.630,7.630,6.188,6.188,5.190,5.190,5.056,5.056]
nEta = len(dzEta)
etaDim = "cm"

## gap half width
dzGap = 0.2500
dzGapDim = "cm"
nGap = nEta - 1

dzIn = sum(dzEta) + nGap * dzGap

## bottom half width
dxBot = 12.513
dxBotDim = "cm"

## top half width
dxTop = 20.997
dxTopDim = "cm"

## slope [cm]
slope = (dxTop - dxBot) / (2.*dzIn)

## distance of centre of chamber to bottom
dzTot = 50.820
dzTotDim = "cm"

# distance from beam-line [cm]
dBeamLine = 1338
dBeamLineDim = "mm"
rPos = dBeamLine + dzTot

##GE1/1 
rMin = 1.3
rMinDim = "m"
rMax = 2.51
rMaxDim = "m"
dz = 4.5
dzDim = "cm"

## gas 
ga_dy = [1.050,0.500,1.600]
ga_dy_dim = "mm"
ga_dx = 0.5
ga_dx_dim = "mm"
