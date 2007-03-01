C
C SF: all common block names have now ME prepended
C
c====================================================================
c
c  Define common block containing all coupling constants and masses.
c
c====================================================================
c
      double complex       gg(2)
      double precision     alpha,ee, sin2w, gfermi, alfas,g
      common/MECOUPL_BASIC/alpha,ee, sin2w, gfermi, alfas,g,gg   
c
      double precision     hmass, wmass, zmass, amass,
     &                     tmass, bmass, lmass, cmass
      common/MECOUPL_MASS/ hmass, wmass, zmass, amass,
     &                     tmass, bmass, lmass, cmass

      double precision     hwidth, wwidth, zwidth, 
     &                     twidth, lwidth, awidth
      common/MECOUPL_WIDTH/hwidth, wwidth, zwidth, 
     &                     twidth, lwidth, awidth

      double complex       gal(2), gad(2), gau(2), gwf(2),
     &                     gzn(2), gzl(2), gzd(2), gzu(2)
      double precision     gw, gwwa, gwwz
      common/MECOUPL_GAUGE/gal   , gad   , gau   , gwf   ,
     &                     gzn   , gzl   , gzd   , gzu   ,
     &                     gw, gwwa, gwwz

      double complex       gwfc(2),  gwfs(2)
      common/MEcoupl_ckm/  gwfc,     gwfs	

      double complex       gwwh, gzzh, gwwhh, gzzhh, ghhh, ghhhh
      common/MECOUPL_SCAL/ gwwh, gzzh, gwwhh, gzzhh, ghhh, ghhhh

      double complex       ghtop(2), ghbot(2), ghtau(2), ghcha(2)
      common/MECOUPL_YUK/  ghtop   , ghbot   , ghtau   , ghcha

      double precision     xzmass, xwmass
      common/MECOUPL_XMASS/xzmass, xwmass

      double complex       xzl(2) , xzb(2) , xzt(2) ,
     &                     xwpq(2), xwmq(2), xwpl(2), xwml(2)
      common/MECOUPL_XFFV/ xzl    , xzb    , xzt    ,
     &                     xwpq   , xwmq   , xwpl   , xwml

      double complex       xzhz, xwhwp, xwhwm
      common/MECOUPL_XVSS/ xzhz, xwhwp, xwhwm

      double complex       xwzwp, xwzwm, xwawp, xwawm
      common/MECOUPL_XVVS/ xwzwp, xwzwm, xwawp, xwawm

      double complex       xwzhwp, xwzhwm, xwahwp, xwahwm
      common/MECOUPL_XVVSS/xwzhwp, xwzhwm, xwahwp, xwahwm
