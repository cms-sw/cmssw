      subroutine urqmd(n)
      include 'epos.inc'

      print *,n
      if(iurqmd.eq.1)stop'compile with eposu.f instead of eposu_no.f '

      end
