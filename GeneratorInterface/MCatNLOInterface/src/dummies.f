      subroutine elpdf_user(qstar2,x,fx,nf)
      dimension fx(-nf:nf)
      return
      end


      subroutine elpdf_dg(qstar2,x,fx,nf)
      dimension fx(-nf:nf)
      return
      end


      subroutine elpdf_lac1(qstar2,x,fx,nf)
      dimension fx(-nf:nf)
      return
      end


      subroutine elpdf_lac2(qstar2,x,fx,nf)
      dimension fx(-nf:nf)
      return
      end


      subroutine elpdf_lac3(qstar2,x,fx,nf)
      dimension fx(-nf:nf)
      return
      end


      subroutine elpdf_gs(qstar2,x,fx,nf)
      dimension fx(-nf:nf)
      return
      end


      subroutine elpdf_grv(qstar2,x,fx,nf)
      dimension fx(-nf:nf)
      return
      end


      subroutine elpdf_acf(qstar2,x,fx,nf)
      dimension fx(-nf:nf)
      return
      end


      subroutine elpdf_afg(qstar2,x,fx,nf)
      dimension fx(-nf:nf)
      return
      end


      subroutine  fxdflm1(x,qstar2,strfun,func)
      character*(*) strfun
      return
      end


      subroutine  fxdflm2(x,qstar2,strfun,func)
      character*(*) strfun
      return
      end


      subroutine  fxdflm3(x,qstar2,strfun,func)
      character*(*) strfun
      return
      end


      function random(iseed)
      random=0
      write(*,*)'This function must not be called'
      stop
      end
