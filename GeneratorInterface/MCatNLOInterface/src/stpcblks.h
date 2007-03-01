c sh = (p_had1 + p_had2)^2 == collider c.m. energy squared
      real * 8 sh
      common/shadr/sh
c xlam = lambda_QCD (5 flavours)
c zg   = strong coupling = sqrt(4*pi*alfas)
c gf   = G_F
c ze2  = electron charge squared
      real * 8 xlam,zg,gf,ze2
      common/fixvar/xlam,zg,gf,ze2
c xm1, xm12 = top quark mass, and its square
c xm2, xm22 = light quark mass (set to zero), and its square
c xmW2      = W mass squared
      real * 8 xm1,xm2,xm12,xm22,xmW2
      common/cmass/xm1,xm2,xm12,xm22,xmW2
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
