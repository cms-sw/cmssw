c----------------------------------------------------------
c Common blocks for the fixed variables
c
c sh = (p_had1 + p_had2)^2
c
c xm0i2 = Pole mass squared of the intermediate vector boson
c gai   = Width of the intermediate vector boson
c xm0v2 = Pole mass squared of the final-state vector boson
c gav   = Width of the final-state vector boson
c xm0h2 = Pole mass squared of the Higgs
c gah   = Width of the Higgs
c xlam = lambda_QCD (5 flavours)
c zg   = strong coupling = sqrt(4*pi*alfas)
c gf   = G_F
c ze2  = electron charge squared
c ze2v = electron charge squared, computed at the vector boson mass
c
      real * 8 sh
      common/shadr/sh
      real * 8 xm0i2,gai,xm0v2,gav,xm0h2,gah,xlam,zg,gf,ze2,ze2v
      common/fixvar/xm0i2,gai,xm0v2,gav,xm0h2,gah,xlam,zg,gf,ze2,ze2v
c xmi2 = Mass squared of the intermediate vector boson
c The other variables are relevant to the Breit-Wigner function
      real * 8 xmi2,bwifmpl,bwifmmn,bwidelf,xmilow2,xmiupp2
      common/bwicmm/xmi2,bwifmpl,bwifmmn,bwidelf,xmilow2,xmiupp2
c xmv2 = Mass squared of the final-state vector boson
c The other variables are relevant to the Breit-Wigner function
      real * 8 xmv2,bwvfmpl,bwvfmmn,bwvdelf,xmvlow2,xmvupp2
      common/bwvcmm/xmv2,bwvfmpl,bwvfmmn,bwvdelf,xmvlow2,xmvupp2
c xmh2 = Mass squared of the Higgs
c The other variables are relevant to the Breit-Wigner function
      real * 8 xmh2,bwhfmpl,bwhfmmn,bwhdelf,xmhlow2,xmhupp2
      common/bwhcmm/xmh2,bwhfmpl,bwhfmmn,bwhdelf,xmhlow2,xmhupp2
c factors to rescale the factorization scales for hadron 1 and 2, and the
c renormalization scale
      real * 8 xf2h1,xf2h2,xren2
      common/scalef/xf2h1,xf2h2,xren2
c factors to rescale the factorization scales for hadron 1 and 2, and the
c renormalization scale (MC terms)
      real * 8 xf2h1mc,xf2h2mc,xren2mc
      common/scalemcf/xf2h1mc,xf2h2mc,xren2mc
c renormalization scale, factorization scales for hadron 1 and 2
      real * 8 xmur2,xmuf2h1,xmuf2h2
      common/phosca/xmur2,xmuf2h1,xmuf2h2
c renormalization scale, factorization scales for hadron 1 and 2 (MC terms)
      real * 8 xmumcr2,xmumcf2h1,xmumcf2h2
      common/phoscamc/xmumcr2,xmumcf2h1,xmumcf2h2
c Number of light flavours (3 for charm, 4 for bottom, 5 for top)
      integer nl
      common/nl/nl
c----------------------------------------------------------
c Flag to select subprocess: prc = 'gg','qq' or 'qg'
c
      character * 2 prc
      common/process/prc
c
c
c scheme = 'DI' for deep inelastic,  'MS' for msbar scheme
      character * 2 schhad1,schhad2
      common/scheme/schhad1,schhad2
c
c
c use newver='NEW' for vaxes, 'UNKNOWN' in other machines,
      character * 7 newver
      common/newver/newver
