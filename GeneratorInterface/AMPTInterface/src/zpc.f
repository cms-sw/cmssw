c.................... zpc.f
c	PROGRAM ZPC
      SUBROUTINE ZPCMN
c       Version: 1.0.1
c       Author: Bin Zhang 
c       (suggestions, problems -> bzhang@nt1.phys.columbia.edu)
cms
cms     dlw & gsfs Comments our writing of output files
cms
        implicit double precision (a-h, o-z)
clin-4/20/01        PARAMETER (NMAXGL = 16000)
        parameter (MAXPTN=400001)
        common /para3/ nsevt, nevnt, nsbrun, ievt, isbrun
cc      SAVE /para3/
        SAVE   
c
c       loop over events
        do 1000 i = 1, nevnt
           ievt = i
c       generation of the initial condition for one event
           call inievt
c      loop over many runs of the same event
           do 2000 j = 1, nsbrun
              isbrun = j
c       initialization for one run of an event
              call inirun
clin-4/2008 not used:
c             CALL HJAN1A
 3000         continue
c       do one collision
              call zpcrun(*4000)
              call zpca1
              goto 3000
 4000         continue
              call zpca2
 2000      continue
 1000   continue
        call zpcou
clin-5/2009 ctest off
c     5/17/01 calculate v2 for parton already frozen out:
c        call flowp(3)
c.....to get average values for different strings
        CALL zpstrg
        RETURN
        end

******************************************************************************
******************************************************************************

        block data zpcbdt
c       set initial values in block data

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        PARAMETER (MAXSTR=150001)
        common /para1/ mul
cc      SAVE /para1/
        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para3/ nsevt, nevnt, nsbrun, ievt, isbrun
cc      SAVE /para3/
        common /para4/ iftflg, ireflg, igeflg, ibstfg
cc      SAVE /para4/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /para6/ centy
cc      SAVE /para6/
clin-6/2009 nsmbbbar and nsmmeson respectively give the total number of 
c     baryons/anti-baryons and mesons for each event:
c        common /para7/ ioscar
        common /para7/ ioscar,nsmbbbar,nsmmeson
cc      SAVE /para7/
        COMMON /prec1/GX0(MAXPTN),GY0(MAXPTN),GZ0(MAXPTN),FT0(MAXPTN),
     &       PX0(MAXPTN), PY0(MAXPTN), PZ0(MAXPTN), E0(MAXPTN),
     &       XMASS0(MAXPTN), ITYP0(MAXPTN)
cc      SAVE /prec1/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec3/gxs(MAXPTN),gys(MAXPTN),gzs(MAXPTN),fts(MAXPTN),
     &     pxs(MAXPTN), pys(MAXPTN), pzs(MAXPTN), es(MAXPTN),
     &     xmasss(MAXPTN), ityps(MAXPTN)
cc      SAVE /prec3/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /prec5/ eta(MAXPTN), rap(MAXPTN), tau(MAXPTN)
cc      SAVE /prec5/
        common /prec6/ etas(MAXPTN), raps(MAXPTN), taus(MAXPTN)
cc      SAVE /prec6/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist2/ icell, icel(10,10,10)
cc      SAVE /ilist2/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
c     6/07/02 initialize in ftime to expedite compiling:
c        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        common /ilist6/ t, iopern, icolln
cc      SAVE /ilist6/
        COMMON /ilist7/ LSTRG0(MAXPTN), LPART0(MAXPTN)
cc      SAVE /ilist7/
        COMMON /ilist8/ LSTRG1(MAXPTN), LPART1(MAXPTN)
cc      SAVE /ilist8/
        common /rndm1/ number
cc      SAVE /rndm1/
        common /rndm2/ iff
cc      SAVE /rndm2/
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        common /ana1/ ts(12)
cc      SAVE /ana1/
        common /ana2/
     &     det(12), dn(12), detdy(12), detdn(12), dndy(12),
     &     det1(12), dn1(12), detdy1(12), detdn1(12), dndy1(12),
     &     det2(12), dn2(12), detdy2(12), detdn2(12), dndy2(12)
cc      SAVE /ana2/
        common /ana3/ em(4, 4, 12)
cc      SAVE /ana3/
        common /ana4/ fdetdy(24), fdndy(24), fdndpt(12)
cc      SAVE /ana4/
        SAVE   
        data centy/0d0/
c     6/07/02 initialize in ftime to expedite compiling:
c        data (ct(i), i = 1, MAXPTN)/MAXPTN*0d0/
c        data (ot(i), i = 1, MAXPTN)/MAXPTN*0d0/
c        data tlarge/1000000.d0/
        data number/0/
        data ts/0.11d0, 0.12d0, 0.15d0, 0.2d0, 0.3d0, 0.4d0, 0.6d0,
     &     0.8d0, 1d0, 2d0, 4d0, 6d0/
c
        end

******************************************************************************
******************************************************************************

        subroutine inizpc

        implicit double precision (a-h, o-z)
        SAVE   

        call readpa

        call inipar

        call inian1

        return
        end

        subroutine readpa

        implicit double precision (a-h, o-z)

cc        external ran1

        character*50 str

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para3/ nsevt, nevnt, nsbrun, ievt, isbrun
cc      SAVE /para3/
        common /para4/ iftflg, ireflg, igeflg, ibstfg
cc      SAVE /para4/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /para7/ ioscar,nsmbbbar,nsmmeson
cc      SAVE /para7/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /rndm1/ number
cc      SAVE /rndm1/
        common /rndm2/ iff
cc      SAVE /rndm2/
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        SAVE   

        str=str
        iseed=iseedp
c       this is the initialization file containing the initial values of 
c          the parameters
cbz1/31/99
c        open (5, file = 'zpc.ini', status = 'unknown')
cbz1/31/99end

c       this is the final data file containing general info about the cascade
cbz1/31/99
c        open (6, file = 'zpc.res', status = 'unknown')
cms     open (25, file = 'ana/zpc.res', status = 'unknown')
cbz1/31/99end

c       this is the input file containing initial particle records
cbz1/25/99
c        open (7, file = 'zpc.inp', status = 'unknown')
cbz1/25/99end

c       this gives the optional OSCAR standard output
cbz1/31/99
c        open (8, file = 'zpc.oscar', status = 'unknown')
        if(ioscar.eq.1) then
cms        open (26, file = 'ana/parton.oscar', status = 'unknown')
cms        open (19, file = 'ana/hadron.oscar', status = 'unknown')
        endif
cbz1/31/99end

c     2/11/03 combine zpc initialization into ampt.ini:
c        open (29, file = 'zpc.ini', status = 'unknown')
c        read (29, *) str, xmp
        xmp=0d0
c        read (29, *) str, xmu
c        read (29, *) str, alpha
        cutof2 = 4.5d0 * (alpha / xmu) ** 2
c        read (29, *) str, rscut2
        rscut2=0.01d0
c        read (29, *) str, nsevt
        nsevt=1
c        read (29, *) str, nevnt
        nevnt=1
c        read (29, *) str, nsbrun
        nsbrun=1
c        read (29, *) str, iftflg
        iftflg=0
c        read (29, *) str, ireflg
        ireflg=1
cbz1/31/99
        IF (ireflg .EQ. 0) THEN
cms        OPEN (27, FILE = 'zpc.inp', STATUS = 'UNKNOWN')
        END IF
cbz1/31/99end
c        read (29, *) str, igeflg
        igeflg=0
c        read (29, *) str, ibstfg
        ibstfg=0
c        read (29, *) str, iconfg
        iconfg=1
c        read (29, *) str, iordsc
        iordsc=11
c        read (29, *) str, ioscar
c        read (29, *) str, v1, v2, v3
        v1=0.2d0
        v2=0.2d0
        v3=0.2d0
c        read (29, *) str, size1, size2, size3
        size1=1.5d0
        size2=1.5d0
        size3=0.7d0
        if (size1 .eq. 0d0 .or. size2 .eq. 0d0 .or. 
     &     size3 .eq. 0d0) then
           if (size1 .ne. 0d0 .or. size2 .ne. 0d0 .or. size3 .ne. 0d0
     &        .or. v1 .ne. 0d0 .or. v2 .ne. 0d0 .or. v3 .ne. 0d0) then
              print *, 'to get rid of space division:'
              print *, 'set all sizes and vs to 0'
              stop 'chker'
           end if
        end if
        size = min(size1, size2, size3)
c        read (29, *) str, iff
        iff=-1
c        read (29, *) str, iseed

c     10/24/02 get rid of argument usage mismatch in ran1():
        isedng=-iseed
c        a = ran1(-iseed)
        a = ran1(isedng)
c        read (29, *) str, irused
        irused=2
        do 1001 i = 1, irused - 1
c           a = ran1(2)
           iseed2=2
           a = ran1(iseed2)
 1001   continue
c     10/24/02-end

        if (iconfg .eq. 2 .or. iconfg .eq. 3) then
           v1 = 0d0
           v2 = 0d0
        end if

        if (iconfg .eq. 4 .or. iconfg .eq. 5) then
           v1 = 0d0
           v2 = 0d0
           v3 = 0d0
        end if

        close(5)

        return
        end

        subroutine inipar

        implicit double precision (a-h,o-z)

        common /para4/ iftflg, ireflg, igeflg, ibstfg
cc      SAVE /para4/
        common /para6/ centy
cc      SAVE /para6/
        SAVE   

        if (ibstfg .ne. 0) then
           centy = -6d0
        end if

        return
        end

        subroutine inian1

        implicit double precision (a-h,o-z)

        common /para4/ iftflg, ireflg, igeflg, ibstfg
cc      SAVE /para4/
        common /ana1/ ts(12)
cc      SAVE /ana1/   
        SAVE   
        if (ibstfg .ne. 0) then
           a = cosh(6d0)
           do 1001 i = 1, 12
              ts(i) = ts(i) * a
 1001      continue
        end if

        return
        end

******************************************************************************

        subroutine inievt

        implicit double precision (a-h, o-z)

        COMMON /para1/ mul
cc      SAVE /para1/
        common /para4/ iftflg, ireflg, igeflg, ibstfg
cc      SAVE /para4/
        SAVE   

cbz1/25/99
c        mul = 0
cbz1/25/99
        if (ireflg .eq. 0) call readi
        if (igeflg .ne. 0) call genei
        if (ibstfg .ne. 0) call boosti

        return
        end

        subroutine readi

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        double precision field(9)
        common /para1/ mul
cc      SAVE /para1/
        common /para3/ nsevt, nevnt, nsbrun, ievt, isbrun
cc      SAVE /para3/
        COMMON /prec1/GX0(MAXPTN),GY0(MAXPTN),GZ0(MAXPTN),FT0(MAXPTN),
     &       PX0(MAXPTN), PY0(MAXPTN), PZ0(MAXPTN), E0(MAXPTN),
     &       XMASS0(MAXPTN), ITYP0(MAXPTN)
cc      SAVE /prec1/
        SAVE   
        do 1001 i = 1, MAXPTN
           if (ievt .ne. 1 .and. i .eq. 1) then
              ityp0(i) = ntyp
              gx0(1) = field(1)
              gy0(1) = field(2)
              gz0(1) = field(3)
              ft0(1) = field(4)
              px0(1) = field(5)
              py0(1) = field(6)
              pz0(1) = field(7)
              e0(1) = field(8)
              xmass0(i) = field(9)
              mul = 1
           else
 900              read (27, *, end = 1000) neve, ntyp, field
              if (neve .lt. nsevt) goto 900
              if (neve .gt.
     &           nsevt + ievt - 1) goto 1000
              ityp0(i) = ntyp
              gx0(i) = field(1)
              gy0(i) = field(2)
              gz0(i) = field(3)
              ft0(i) = field(4)
              px0(i) = field(5)
              py0(i) = field(6)
              pz0(i) = field(7)
              e0(i) = field(8)
              xmass0(i) = field(9)
              mul = mul + 1
           end if
 1001   continue
        
 1000        continue
        
        return
        end

        subroutine genei

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /para1/ mul
cc      SAVE /para1/
        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para3/ nsevt, nevnt, nsbrun, ievt, isbrun
cc      SAVE /para3/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        COMMON /prec1/GX0(MAXPTN),GY0(MAXPTN),GZ0(MAXPTN),FT0(MAXPTN),
     &       PX0(MAXPTN), PY0(MAXPTN), PZ0(MAXPTN), E0(MAXPTN),
     &       XMASS0(MAXPTN), ITYP0(MAXPTN)
cc      SAVE /prec1/
        common /prec5/ eta(MAXPTN), rap(MAXPTN), tau(MAXPTN)
cc      SAVE /prec5/
        common /lor/ enenew, pxnew, pynew, pznew
cc      SAVE /lor/
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        SAVE   

cc        external ran1

        iseed=iseedp
        incmul = 4000
        temp = 0.5d0
        etamin = -5d0        
        etamax = 5d0
        r0 = 5d0
        tau0 = 0.1d0
        deta = etamax - etamin

        do 1001 i = mul + 1, mul + incmul
           ityp0(i) = 21
           xmass0(i) = xmp
           call energy(e, temp)
           call momntm(px, py, pz, e)
c     7/20/01:
c           e = sqrt(e ** 2 + xmp ** 2)
           e = dsqrt(e ** 2 + xmp ** 2)
           if (iconfg .le. 3) then
              eta(i) = etamin + deta * ran1(iseed)
              bex = 0d0
              bey = 0d0
              bez = -tanh(eta(i))
              call lorenz(e, px, py, pz, bex, bey, bez)
              px0(i) = pxnew
              py0(i) = pynew
              pz0(i) = pznew
              e0(i) = enenew
           else
              px0(i) = px
              py0(i) = py
              pz0(i) = pz
              e0(i) = e
           end if
 1001   continue

        do 1002 i = mul + 1, mul + incmul
           if (iconfg .le. 3) then
              gz0(i) = tau0 * sinh(eta(i))
              ft0(i) = tau0 * cosh(eta(i))
              if (iconfg .eq. 1) then
                 call posit1(x, y, r0)
                 gx0(i) = x + px0(i) * ft0(i)/e0(i)
                 gy0(i) = y + py0(i) * ft0(i)/e0(i)
              else if (iconfg .eq. 2 .or. iconfg .eq. 3) then
                 call posit2(x, y)
                 gx0(i) = x
                 gy0(i) = y
              end if
           else
              ft0(i) = 0d0
              call posit3(x, y, z)
              gx0(i) = x
              gy0(i) = y
              gz0(i) = z
           end if
 1002   continue

        mul = mul + incmul
            
c       check if it's necessary to adjust array size 'adarr'
            if (mul .ge. MAXPTN .or. mul .eq. 0) then
           print *, 'event',ievt,'has',mul,'number of gluon',
     &          'adjusting counting is necessary'
           stop 'adarr'
        end if
        
        return
        end

        subroutine posit1(x, y, r0)
        
        implicit double precision (a-h, o-z)

cc        external ran1
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        SAVE   

        iseed=iseedp
 10        x = 2d0 * ran1(iseed) - 1d0
        y = 2d0 * ran1(iseed) - 1d0
        if (x ** 2 + y ** 2 .gt. 1d0) goto 10
        x = x * r0
        y = y * r0
        
        return
        end

        subroutine posit2(x, y)
        
        implicit double precision (a-h, o-z)

c        external ran1

        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        SAVE   
        iseed=iseedp
         x = 2d0 * ran1(iseed) - 1d0
        y = 2d0 * ran1(iseed) - 1d0
        x = x * 5d0 * size1
        y = y * 5d0 * size2
        
        return
        end

        subroutine posit3(x, y, z)
        
        implicit double precision (a-h, o-z)

cc        external ran1

        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        SAVE   

        iseed=iseedp
         x = 2d0 * ran1(iseed) - 1d0
        y = 2d0 * ran1(iseed) - 1d0
        z = 2d0 * ran1(iseed) - 1d0
        x = x * 5d0 * size1
        y = y * 5d0 * size2
        z = z * 5d0 * size3
        
        return
        end
        
        subroutine energy(e, temp)

c       to generate the magnitude of the momentum e,
c       knowing the temperature of the local thermal distribution temp
        
        implicit double precision (a-h, o-z)
        
cc        external ran1

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        SAVE   

        iseed=iseedp
 1000        continue
        
        e = ran1(iseed)
        e = e * ran1(iseed)
        e = e * ran1(iseed)

        if (e .le. 0d0) goto 1000
        e = - temp * log(e)
        if (ran1(iseed) .gt. 
     &     exp((e - dsqrt(e ** 2 + xmp ** 2))/temp)) then
           goto 1000
        end if

        return
        end
        
        subroutine momntm(px, py, pz, e)

c       to generate the 3 components of the momentum px, py, pz,
c       from the magnitude of the momentum e
        
        implicit double precision (a-h,o-z)
        
cc        external ran1
        
        parameter (pi = 3.14159265358979d0)
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        SAVE   

        iseed=iseedp
        cost = 2d0 * ran1(iseed) - 1d0
c     7/20/01:
c        sint = sqrt(1d0 - cost ** 2)
        sint = dsqrt(1d0 - cost ** 2)
        phi = 2d0 * pi * ran1(iseed)
      
        px = e * sint * cos(phi)
        py = e * sint * sin(phi)
        pz = e * cost
        
        return
        end

        subroutine boosti

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /para1/ mul
cc      SAVE /para1/
        common /para6/ centy
cc      SAVE /para6/
        COMMON /prec1/GX0(MAXPTN),GY0(MAXPTN),GZ0(MAXPTN),FT0(MAXPTN),
     &       PX0(MAXPTN), PY0(MAXPTN), PZ0(MAXPTN), E0(MAXPTN),
     &       XMASS0(MAXPTN), ITYP0(MAXPTN)
cc      SAVE /prec1/
        common /lor/ enenew, pxnew, pynew, pznew
cc      SAVE /lor/
        SAVE   

        external lorenz

        bex = 0d0 
        bey = 0d0
        bez = - tanh(centy)
        
c       save data for many runs of the same initial condition           
        do 1001 i = 1, mul
           px1 = gx0(i)
           py1 = gy0(i)
           pz1 = gz0(i)
           e1 = ft0(i)
           call lorenz(e1, px1, py1, pz1, bex, bey, bez)
           gx0(i) = pxnew
           gy0(i) = pynew
           gz0(i) = pznew
           ft0(i) = enenew
           px1 = px0(i)
           py1 = py0(i)
           pz1 = pz0(i)
           e1 = e0(i)
           call lorenz(e1, px1, py1, pz1, bex, bey, bez)
           px0(i) = pxnew
           py0(i) = pynew
           pz0(i) = pznew
           e0(i) = enenew
 1001   continue
        
        return
        end

******************************************************************************

        subroutine inirun
        SAVE   

c       sort prec2 according to increasing formation time
        call ftime
        call inirec
        call iilist
        call inian2

        return
        end

        subroutine ftime
c       this subroutine generates formation time for the particles
c       indexing ft(i)
c       input e(i)
c       output ft(i), indx(i)

        implicit double precision (a-h, o-z)

        external ftime1
        parameter (MAXPTN=400001)
        common /para1/ mul
cc      SAVE /para1/
        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para4/ iftflg, ireflg, igeflg, ibstfg
cc      SAVE /para4/
        COMMON /prec1/GX0(MAXPTN),GY0(MAXPTN),GZ0(MAXPTN),FT0(MAXPTN),
     &       PX0(MAXPTN), PY0(MAXPTN), PZ0(MAXPTN), E0(MAXPTN),
     &       XMASS0(MAXPTN), ITYP0(MAXPTN)
cc      SAVE /prec1/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        common /par1/ formt
cc      SAVE /par1/
        common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        SAVE   

        iseed=iseedp
clin-6/07/02 initialize here to expedite compiling, instead in zpcbdt:
        do 1001 i = 1, MAXPTN
           ct(i)=0d0
           ot(i)=0d0
 1001   continue
        tlarge=1000000.d0
clin-6/07/02-end

        if (iftflg .eq. 0) then
c     5/01/01 different prescription for parton initial formation time:
           if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) then
              do 1002 i = 1, mul
                 if (ft0(i) .gt. tlarge) ft0(i) = tlarge
 1002         continue
              goto 150
           else
c     5/01/01-end

           do 1003 i = 1, MAXPTN
              ft0(i) = tlarge
 1003      continue
           do 1004 i = 1, mul
              xmt2 = px0(i) ** 2 + py0(i) ** 2 + xmp ** 2
              formt = xmt2 / e0(i)           
              ft0(i) = ftime1(iseed)
              if (ft0(i) .gt. tlarge) ft0(i) = tlarge
 1004      continue
c     5/01/01:
        endif

        end if

c     5/01/01:
 150        continue

c        call index1(MAXPTN, mul, ft0, indx)
        if (mul .gt. 1) then
           call index1(MAXPTN, mul, ft0, indx)
        else
clin-7/09/03: need to set value for mul=1:
           indx(1)=1
        end if
c
        return
        end

        subroutine inirec

        implicit double precision (a-h, o-z)
cc        external ran1
        parameter (MAXPTN=400001)
        common /para1/ mul
cc      SAVE /para1/
        common /para4/ iftflg, ireflg, igeflg, ibstfg
cc      SAVE /para4/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        COMMON /prec1/GX0(MAXPTN),GY0(MAXPTN),GZ0(MAXPTN),FT0(MAXPTN),
     &       PX0(MAXPTN), PY0(MAXPTN), PZ0(MAXPTN), E0(MAXPTN),
     &       XMASS0(MAXPTN), ITYP0(MAXPTN)
cc      SAVE /prec1/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec3/gxs(MAXPTN),gys(MAXPTN),gzs(MAXPTN),fts(MAXPTN),
     &     pxs(MAXPTN), pys(MAXPTN), pzs(MAXPTN), es(MAXPTN),
     &     xmasss(MAXPTN), ityps(MAXPTN)
cc      SAVE /prec3/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /prec5/ eta(MAXPTN), rap(MAXPTN), tau(MAXPTN)
cc      SAVE /prec5/
        common /prec6/ etas(MAXPTN), raps(MAXPTN), taus(MAXPTN)
cc      SAVE /prec6/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
cbz1/25/99
        COMMON /ilist7/ LSTRG0(MAXPTN), LPART0(MAXPTN)
cc      SAVE /ilist7/
        COMMON /ilist8/ LSTRG1(MAXPTN), LPART1(MAXPTN)
cc      SAVE /ilist8/
cbz1/25/99end
        COMMON /smearz/smearp,smearh
cc      SAVE /smearz/
        dimension vxp(MAXPTN), vyp(MAXPTN), vzp(MAXPTN)
        common /precpa/ vxp0(MAXPTN), vyp0(MAXPTN), vzp0(MAXPTN)
cc      SAVE /precpa/
        common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
clin-6/06/02 local parton freezeout:
        common /frzprc/ 
     &       gxfrz(MAXPTN), gyfrz(MAXPTN), gzfrz(MAXPTN), ftfrz(MAXPTN),
     &       pxfrz(MAXPTN), pyfrz(MAXPTN), pzfrz(MAXPTN), efrz(MAXPTN),
     &       xmfrz(MAXPTN), 
     &       tfrz(302), ifrz(MAXPTN), idfrz(MAXPTN), itlast
cc      SAVE /frzprc/
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        common /para7/ ioscar,nsmbbbar,nsmmeson
        COMMON /AREVT/ IAEVT, IARUN, MISS
        SAVE   
        iseed=iseedp
clin-6/06/02 local freezeout initialization:
        if(isoft.eq.5) then
           itlast=0
           call inifrz
        endif

        do 1001 i = 1, mul
clin-7/09/01 define indx(i) to save time:
c           ityp(i) = ityp0(indx(i))
c           gx(i) = gx0(indx(i))
c           gy(i) = gy0(indx(i))
c           gz(i) = gz0(indx(i))
c           ft(i) = ft0(indx(i))
c           px(i) = px0(indx(i))
c           py(i) = py0(indx(i))
c           pz(i) = pz0(indx(i))
c           e(i) = e0(indx(i))
c           xmass(i) = xmass0(indx(i))
ccbz1/25/99
c           LSTRG1(I) = LSTRG0(INDX(I))
c           LPART1(I) = LPART0(INDX(I))
ccbz1/25/99end
           indxi=indx(i)
           ityp(i) = ityp0(indxi)
           gx(i) = gx0(indxi)
           gy(i) = gy0(indxi)
           gz(i) = gz0(indxi)
           ft(i) = ft0(indxi)
           px(i) = px0(indxi)
           py(i) = py0(indxi)
           pz(i) = pz0(indxi)
           e(i) = e0(indxi)
           xmass(i) = xmass0(indxi)
           LSTRG1(I) = LSTRG0(INDXI)
           LPART1(I) = LPART0(INDXI)
           vxp(I) = vxp0(INDXI)
           vyp(I) = vyp0(INDXI)
           vzp(I) = vzp0(INDXI)
clin-7/09/01-end
c
clin-6/06/02 local freezeout initialization:
         if(isoft.eq.5) then
            idfrz(i)=ityp(i)
            gxfrz(i)=gx(i)
            gyfrz(i)=gy(i)
            gzfrz(i)=gz(i)
            ftfrz(i)=ft(i)
            pxfrz(i)=px(i)
            pyfrz(i)=py(i)
            pzfrz(i)=pz(i)
            efrz(i)=e(i)
            xmfrz(i)=xmass(i)
            ifrz(i)=0
         endif
clin-6/06/02-end
 1001 continue

c       save particle info for fixed time analysis
        do 1002 i = 1, mul
           ityps(i) = ityp(i)
           gxs(i) = gx(i)
           gys(i) = gy(i)
           gzs(i) = gz(i)
           fts(i) = ft(i)
           pxs(i) = px(i)
           pys(i) = py(i)
           pzs(i) = pz(i)
           es(i) = e(i)
           xmasss(i) = xmass(i)
 1002   continue

clin-6/2009
cms     if(isoft.eq.1.and.(ioscar.eq.2.or.ioscar.eq.3))
cms  1       write(92,*) iaevt,miss,mul

        do 1003 i = 1, mul
           energy = e(i)
           vx(i) = px(i) / energy
           vy(i) = py(i) / energy
           vz(i) = pz(i) / energy
           if (iftflg .eq. 0) then
              formt = ft(i)
c     7/09/01 propagate partons with parent velocity till formation
c     so that partons in same hadron have 0 distance:
c            gx(i) = gx(i) + vx(i) * formt
c            gy(i) = gy(i) + vy(i) * formt
c            gz(i) = gz(i) + vz(i) * formt
            if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) then
               gx(i) = gx(i) + vxp(i) * formt
               gy(i) = gy(i) + vyp(i) * formt
               gz(i) = gz(i) + vzp(i) * formt
            else
               gx(i) = gx(i) + vx(i) * formt
               gy(i) = gy(i) + vy(i) * formt
               gz(i) = gz(i) + vz(i) * formt
            endif
c     7/09/01-end
c
c     3/27/00-ctest off no smear z on partons to avoid eta overflow:
c              gz(i) = gz(i)+smearp*(2d0 * ran1(iseed) - 1d0)
c     to give eta=y +- smearp*random:
c              smeary=smearp*(2d0 * ran1(iseed) - 1d0)
c              smearf=dexp(2*smeary)*(1+vz(i))/(1-vz(i)+1.d-8)
c              gz(i) = gz(i)+formt*(smearf-1)/(smearf+1)
c     3/27/00-end
           end if

clin-6/2009 write out initial parton information after string melting
c     and after propagating to its format time:
           if(ioscar.eq.2.or.ioscar.eq.3) then
              if(dmax1(abs(gx(i)),abs(gy(i)),
     1             abs(gz(i)),abs(ft(i))).lt.9999) then
cms              write(92,200) ityp(i),px(i),py(i),pz(i),xmass(i),
cms  1                gx(i),gy(i),gz(i),ft(i)
              else
cms              write(92,201) ityp(i),px(i),py(i),pz(i),xmass(i),
cms  1                gx(i),gy(i),gz(i),ft(i)
              endif
           endif
cyy 200      format(I6,2(1x,f8.3),1x,f10.3,1x,f6.3,4(1x,f8.2))
cyy 201      format(I6,2(1x,f8.3),1x,f10.3,1x,f6.3,4(1x,e8.2))
c
 1003   continue

        if (iconfg .le. 3) then
           do 1004 i = 1, mul
              if (ft(i) .le. abs(gz(i))) then
                 eta(i) = 1000000.d0
              else
                 eta(i) = 0.5d0 * log((ft(i) + gz(i)) / (ft(i) - gz(i)))
              end if
              if (e(i) .le. abs(pz(i))) then
                 rap(i) = 1000000.d0
              else
                 rap(i) = 0.5d0 * log((e(i) + pz(i)) / (e(i) - pz(i)))
              end if
              tau(i) = ft(i) / cosh(eta(i))
 1004      continue
           
           do 1005 i = 1, mul
              etas(i) = eta(i)
              raps(i) = rap(i)
              taus(i) = tau(i)
 1005      continue
        end if

        return
        end

        subroutine iilist

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /para1/ mul
cc      SAVE /para1/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist2/ icell, icel(10,10,10)
cc      SAVE /ilist2/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        common /ilist6/ t, iopern, icolln
cc      SAVE /ilist6/
        SAVE   

        iscat = MAXPTN
        jscat = MAXPTN

        do 1001 i = 1, mul
           next(i) = 0
           last(i) = 0
           icsta(i) = 0
           nic(i) = 0
           icels(i) = 0
 1001   continue

        icell = 0
        do 1004 i1 = 1, 10
           do 1003 i2 = 1, 10
              do 1002 i3 = 1, 10
                 icel(i1, i2, i3) = 0
 1002         continue
 1003      continue
 1004   continue

        ichkpt = 0
        ifmpt = 1

        do 1005 i = 1, mul
           ct(i) = tlarge
           ot(i) = tlarge
 1005   continue

        iopern = 0
        icolln = 0
        t = 0.d0

        return
        end

        subroutine inian2

        implicit double precision (a-h, o-z)

        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /ana2/
     &     det(12), dn(12), detdy(12), detdn(12), dndy(12),
     &     det1(12), dn1(12), detdy1(12), detdn1(12), dndy1(12),
     &     det2(12), dn2(12), detdy2(12), detdn2(12), dndy2(12)
cc      SAVE /ana2/
        SAVE   

        if (iconfg .le. 3) then
           do 1001 i = 1, 12
              det(i) = 0d0
              dn(i) = 0d0
              det1(i) = 0d0
              dn1(i) = 0d0
              det2(i) = 0d0
              dn2(i) = 0d0
 1001      continue
        end if

        return
        end

******************************************************************************
******************************************************************************

        subroutine zpcrun(*)

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        parameter (tend1 = 250d0)
        parameter (tend2 = 6.1d0)
        common /para1/ mul
cc      SAVE /para1/
        common /para5/ iconfg, iordsc
        common /para7/ ioscar,nsmbbbar,nsmmeson
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /prec5/ eta(MAXPTN), rap(MAXPTN), tau(MAXPTN)
cc      SAVE /prec5/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        common /ilist6/ t, iopern, icolln
cc      SAVE /ilist6/
        common /ana1/ ts(12)
cc      SAVE /ana1/
        common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
        COMMON /AREVT/ IAEVT, IARUN, MISS
        SAVE   

c       save last collision info
        if (mod(ictype, 2) .eq. 0) then
           call savrec(iscat)
           call savrec(jscat)
        end if

c1      get operation type
        call getict(t1)
c2      check freezeout condition
        if (iconfg .eq. 1 .and. t1 .gt. tlarge / 2d0) return 1
        if (iconfg .eq. 2 .or. iconfg .eq. 3) then
           if (t1 .gt. tend1) return 1
c           if (ichkpt .eq. mul) then
c              ii = 0
c              do i = 1, mul
c                 gztemp = gz(i) + vz(i) * (t1 - ft(i))
c                 if (sqrt(t1 ** 2 - gztemp ** 2) .lt. tend) then
c                    ii = 1
c                    goto 1000
c                 end if
c              end do
c 1000              continue
c              if (ii .eq. 0) return 1
c           end if
        end if
        if (iconfg .eq. 4 .or. iconfg .eq. 5) then
           if (t1 .gt. tend2) return 1
        end if

clin-6/06/02 local freezeout for string melting,
c     decide what partons have frozen out at time t1:
      if(isoft.eq.5) then
         call local(t1)
      endif

c3      update iopern, t

        iopern = iopern + 1
        t = t1
        if (mod(ictype, 2) .eq. 0) then
           icolln = icolln + 1

c     4/18/01-ctest off
c           write (2006, 1233) 'iscat=', iscat, 'jscat=', jscat,
c           write (2006, *) 'iscat=', iscat, ' jscat=', jscat,
c     1 ityp(iscat), ityp(jscat)
c           write (2006, 1233) 'iscat=', max(indx(iscat), indx(jscat)),
c     &        'jscat=', min(indx(iscat), indx(jscat))

c           write (2006, 1234) ' icolln=', icolln, 't=', t

c 1233           format (a10, i10, a10, i10)
c 1234           format (a15, i10, a5, f23.17, a5, f23.17)
        end if

c4.1    deal with formation
        if (iconfg .eq. 1
     &     .or. iconfg .eq. 2
     &     .or. iconfg .eq. 4) then
           if (ictype .eq. 1 .or. ictype .eq. 2 .or. 
     &        ictype .eq. 5 .or. ictype .eq. 6) then
              call celasn
           end if
        end if

c4.2    deal with collisions

        if (ictype .ne. 1) then

           iscat0 = iscat
           jscat0 = jscat
           
c        iscat is the larger one so that if it's a wall collision,
c       it's still ok
           iscat = max0(iscat0, jscat0)
           jscat = min0(iscat0, jscat0)

ctest off check icsta(i): 0 with f77 compiler
c        write(9,*) 'BB:ictype,t1,iscat,jscat,icsta(i)=',
c     1 ictype,t1,iscat,jscat,icsta(iscat)
           
c       check collision time table error 'tterr'
clin-4/2008 to avoid out-of-bound error in next():
c           if (jscat .ne. 0 .and. next(jscat) .ne. iscat)
c     &        then
c              print *, 'iscat=', iscat, 'jscat=', jscat,
c     &             'next(', jscat, ')=', next(jscat)
c
c              if (ct(iscat) .lt. tlarge / 2d0) stop 'tterr'
c              if (ct(jscat) .lt. tlarge / 2d0) stop 'tterr'
c           end if 
           if (jscat .ne. 0) then
              if(next(jscat) .ne. iscat) then
                 print *, 'iscat=', iscat, 'jscat=', jscat,
     &                'next(', jscat, ')=', next(jscat)
                 if (ct(iscat) .lt. tlarge / 2d0) stop 'tterr'
                 if (ct(jscat) .lt. tlarge / 2d0) stop 'tterr'
              endif
           end if 
clin-4/2008-end
           
c4.2.1     collisions with wall

c     8/19/02 avoid actual argument in common blocks of cellre:
         niscat=iscat
         njscat=jscat
c           if (icsta(iscat) .ne. 0) call cellre(iscat, t)
c           if (jscat .ne. 0) then
c              if (icsta(jscat) .ne. 0) call cellre(jscat, t)
c           end if
           if (icsta(iscat) .ne. 0) call cellre(niscat, t)
           if (jscat .ne. 0) then
              if (icsta(jscat) .ne. 0) call cellre(njscat, t)
           end if

c4.2.2     collision between particles     

clin-6/2009 write out info for each collision:
c           if (mod(ictype, 2) .eq. 0) call scat(t, iscat, jscat)
           if (mod(ictype, 2) .eq. 0) then
              if(ioscar.eq.3) then
cms              write(95,*) 'event,miss,iscat,jscat=',iaevt,miss,iscat,jscat
                 if(dmax1(abs(gx(iscat)),abs(gy(iscat)),
     1                abs(gz(iscat)),abs(ft(iscat)),abs(gx(jscat)),
     2                abs(gy(jscat)),abs(gz(jscat)),abs(ft(jscat)))
     3                .lt.9999) then
cms                 write(95,200) ityp(iscat),px(iscat),py(iscat),
cms  1                   pz(iscat),xmass(iscat),gx(iscat),gy(iscat),
cms  2                   gz(iscat),ft(iscat)
cms                 write(95,200) ityp(jscat),px(jscat),py(jscat),
cms  1                   pz(jscat),xmass(jscat),gx(jscat),gy(jscat),
cms  2                   gz(jscat),ft(jscat)
                 else
cms                 write(95,201) ityp(iscat),px(iscat),py(iscat),
cms  1                   pz(iscat),xmass(iscat),gx(iscat),gy(iscat),
cms  2                   gz(iscat),ft(iscat)
cms                 write(95,201) ityp(jscat),px(jscat),py(jscat),
cms  1                   pz(jscat),xmass(jscat),gx(jscat),gy(jscat),
cms  2                   gz(jscat),ft(jscat)
                 endif
              endif
c     
              call scat(t, iscat, jscat)
c     
              if(ioscar.eq.3) then
                 if(dmax1(abs(gx(iscat)),abs(gy(iscat)),
     1                abs(gz(iscat)),abs(ft(iscat)),abs(gx(jscat)),
     2                abs(gy(jscat)),abs(gz(jscat)),abs(ft(jscat)))
     3                .lt.9999) then
cms                 write(95,200) ityp(iscat),px(iscat),py(iscat),
cms  1                   pz(iscat),xmass(iscat),gx(iscat),gy(iscat),
cms  2                   gz(iscat),ft(iscat)
cms                 write(95,200) ityp(jscat),px(jscat),py(jscat),
cms  1                   pz(jscat),xmass(jscat),gx(jscat),gy(jscat),
cms  2                   gz(jscat),ft(jscat)
                 else
cms                 write(95,201) ityp(iscat),px(iscat),py(iscat),
cms  1                   pz(iscat),xmass(iscat),gx(iscat),gy(iscat),
cms  2                   gz(iscat),ft(iscat)
cms                 write(95,201) ityp(jscat),px(jscat),py(jscat),
cms  1                   pz(jscat),xmass(jscat),gx(jscat),gy(jscat),
cms  2                   gz(jscat),ft(jscat)
                 endif
              endif
           endif
cyy           format(I6,2(1x,f8.3),1x,f10.3,1x,f6.3,4(1x,f8.2))
cyy           format(I6,2(1x,f8.3),1x,f10.3,1x,f6.3,4(1x,e8.2))
           
        end if

c5      update the interaction list
        call ulist(t)

c6      update ifmpt. ichkpt
c       old ichkpt and ifmpt are more conveniently used in ulist
        if (ifmpt .le. mul) then
           if (ictype .ne. 0 .and. ictype .ne. 3 
     &        .and. ictype .ne. 4) then
              ichkpt = ichkpt + 1
              ifmpt = ifmpt + 1
           end if
        end if

        return
        end

        subroutine savrec(i)

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec3/gxs(MAXPTN),gys(MAXPTN),gzs(MAXPTN),fts(MAXPTN),
     &     pxs(MAXPTN), pys(MAXPTN), pzs(MAXPTN), es(MAXPTN),
     &     xmasss(MAXPTN), ityps(MAXPTN)
cc      SAVE /prec3/
        common /prec5/ eta(MAXPTN), rap(MAXPTN), tau(MAXPTN)
cc      SAVE /prec5/
        common /prec6/ etas(MAXPTN), raps(MAXPTN), taus(MAXPTN)
cc      SAVE /prec6/
        SAVE   

        ityps(i) = ityp(i)
        gxs(i) = gx(i)
        gys(i) = gy(i)
        gzs(i) = gz(i)
        fts(i) = ft(i)
        pxs(i) = px(i)
        pys(i) = py(i)
        pzs(i) = pz(i)
        es(i) = e(i)
        xmasss(i) = xmass(i)
        etas(i) = eta(i)
        raps(i) = rap(i)
        taus(i) = tau(i)

        return
        end

        subroutine getict(t1)
        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /para1/ mul
cc      SAVE /para1/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

c       neglect possibility of 2 collisions at the same time
c0       set initial conditions

        t1 = tlarge
        iscat = 0
        jscat = 0

c1      get next collision between particles
        do 1001 i = 1, ichkpt
           if (ot(i) .lt. t1) then
              t1 = ot(i)
              iscat = i
           end if
 1001   continue
        if (iscat .ne. 0) jscat = next(iscat)

c2      get ictype
c     10/30/02 ictype=0:collision; 1:parton formation
        if (iscat .ne. 0 .and. jscat .ne. 0) then
           if (icsta(iscat) .eq. 0 .and. icsta(jscat) .eq. 0) then
              ictype = 0
           else
              ictype = 4
           end if
        else if (iscat .ne. 0 .or. jscat .ne. 0) then
           ictype = 3
        end if
c
        if (ifmpt .le. mul) then
           if (ft(ifmpt) .lt. t1) then
              ictype = 1
              t1 = ft(ifmpt)
           else if (ft(ifmpt) .eq. t1) then
              if (ictype .eq. 0) ictype = 2
              if (ictype .eq. 3) ictype = 5
              if (ictype .eq. 4) ictype = 6
           end if
        end if

        return
        end

        subroutine celasn
c       this subroutine is used to assign a cell for a newly formed particle
c       output: nic(MAXPTN) icels(MAXPTN) in the common /ilist1/
c       icell, and icel(10,10,10) in the common /ilist2/

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /para1/ mul
cc      SAVE /para1/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist2/ icell, icel(10,10,10)
cc      SAVE /ilist2/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        SAVE   

        external integ

        i = ifmpt
        tt = ft(i)
        td = tt - size
        if (iconfg .eq. 1 .and. (size1 .eq. 0d0 .or.
     &     size2 .eq. 0d0 .or. size3 .eq. 0d0)) then
           i1 = 11
           i2 = 11
           i3 = 11
        else if (iconfg .eq. 4 .or. td .le. 0d0) then
           i1 = integ(gx(i) / size1) + 6
           i2 = integ(gy(i) / size2) + 6
           i3 = integ(gz(i) / size3) + 6
           if (integ(gx(i) / size1) .eq. gx(i) / size1 .and. 
     &        vx(i) .lt. 0d0)
     &        i1 = i1 - 1
           if (integ(gy(i) / size2) .eq. gy(i) / size2 .and. 
     &        vy(i) .lt. 0d0)
     &        i2 = i2 - 1
           if (integ(gz(i) / size3) .eq. gz(i) / size3 .and. 
     &        vz(i) .lt. 0d0)
     &        i3 = i3 - 1
        else
           i1 = integ(gx(i) / (size1 + v1 * td)) + 6
           i2 = integ(gy(i) / (size2 + v2 * td)) + 6
           i3 = integ(gz(i) / (size3 + v3 * td)) + 6
           if (integ(gx(i) / (size1 + v1 * td)) .eq. gx(i) / 
     &        (size1 + v1 * td) .and. vx(i) .lt. (i1 - 6) * v1)
     &        i1 = i1 - 1
           if (integ(gy(i) / (size2 + v2 * td)) .eq. gy(i)/
     &        (size2 + v2 * td) .and. vy(i) .lt. (i2 - 6) * v2)
     &        i2 = i2 - 1
           if (integ(gz(i) / (size3 + v3 * td)) .eq. gz(i)/
     &        (size3 + v3 * td) .and. vz(i) .lt. (i3 - 6) * v3)
     &        i3 = i3 - 1
        end if

        if (i1 .le. 0 .or. i1 .ge. 11 .or. i2 .le. 0 .or.
     &     i2 .ge. 11 .or. i3 .le. 0 .or. i3 .ge. 11) then
           i1 = 11
           i2 = 11
           i3 = 11
        end if

        if (i1 .eq. 11) then
           j = icell
           call newcre(i, j)
           icell = j
           icels(i) = 111111
        else
           j = icel(i1, i2, i3)
           call newcre(i, j)
           icel(i1, i2, i3) = j
           icels(i) = i1 * 10000 + i2 * 100 + i3
        end if

        return
        end

        integer function integ(x)
c       this function is used to get the largest integer that is smaller than
c       x

        implicit double precision (a-h, o-z)
        SAVE   

        if (x .lt. 0d0) then
           integ = int(x - 1d0)
        else
           integ = int( x )
        end if

        return
        end

        subroutine cellre(i, t)
c       this subroutine is used for changing the cell of a particle that
c       collide with the wall

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist2/ icell, icel(10,10,10)
cc      SAVE /ilist2/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical good

        external integ

c       this happens before update the /prec2/ common; in contrast with 
c       scat which happens after updating the glue common

        t0 = t

 1000        continue

        if (iconfg .eq. 3 .or. iconfg .eq. 5) then
           k = mod(icsta(i), 10)

           if (k .eq. 1) then
              gx(i) = gx(i) - 10d0 * size1
              dgxa(i) = dgxa(i) + 10d0 * size1
              do 1001 ii = 1, ichkpt
                 if (next(ii) .eq. i) then
                    dgxa(ii) = dgxa(ii) - 10d0 * size1
                 end if
 1001         continue
           end if
           if (k .eq. 2) then
              gx(i) = gx(i) + 10d0 * size1
              dgxa(i) = dgxa(i) - 10d0 * size1
              do 1002 ii = 1, ichkpt
                 if (next(ii) .eq. i) then
                    dgxa(ii) = dgxa(ii) + 10d0 * size1
                 end if
 1002         continue
           end if
           if (k .eq. 3) then
              gy(i) = gy(i) - 10d0 * size2
              dgya(i) = dgya(i) + 10d0 * size2
              do 1003 ii = 1, ichkpt
                 if (next(ii) .eq. i) then
                    dgya(ii) = dgya(ii) - 10d0 * size2
                 end if
 1003         continue
           end if
           if (k .eq. 4) then
              gy(i) = gy(i) + 10d0 * size2
              dgya(i) = dgya(i) - 10d0 * size2
              do 1004 ii = 1, ichkpt
                 if (next(ii) .eq. i) then
                    dgya(ii) = dgya(ii) + 10d0 * size2
                 end if
 1004         continue
           end if
           if (iconfg .eq. 5) then
              if (k .eq. 5) then
                 gz(i) = gz(i) - 10d0 * size3
                 dgza(i) = dgza(i) + 10d0 * size3
                 do 1005 ii = 1, ichkpt
                    if (next(ii) .eq. i) then
                       dgza(ii) = dgza(ii) - 10d0 * size3
                    end if
 1005            continue
              end if
              if (k .eq. 6) then
                 gz(i) = gz(i) + 10d0 * size3
                 dgza(i) = dgza(i) - 10d0 * size3
                 do 1006 ii = 1, ichkpt
                    if (next(ii) .eq. i) then
                       dgza(ii) = dgza(ii) + 10d0 * size3
                    end if
 1006               continue
              end if
           end if
        else
           icels0 = icels(i)

           i1 = icels0 / 10000
           i2 = (icels0 - i1 * 10000) / 100
           i3 = icels0 - i1 * 10000 - i2 * 100
           
cc       for particle inside the cube
           if (i1 .ge. 1 .and. i1 .le. 10
     &        .and. i2 .ge. 1 .and. i2 .le. 10
     &        .and. i3 .ge. 1 .and. i3 .le. 10) then

c       this assignment takes care of nic(i)=0 automatically
              if (icel(i1, i2, i3) .eq. i) icel(i1, i2, i3) = nic(i)

c1      rearrange the old cell

              call oldcre(i)

c2      rearrange the new cell

              k = mod(icsta(i), 10)
              
c2.1    particle goes out of the cube       
              if (iconfg .eq. 1) then
                 good = (i1 .eq. 1 .and. k .eq. 2)
     &              .or. (i1 .eq. 10 .and. k .eq. 1)
     &              .or. (i2 .eq. 1 .and. k .eq. 4)
     &              .or. (i2 .eq. 10 .and. k .eq. 3)
     &              .or. (i3 .eq. 1 .and. k .eq. 6)
     &              .or. (i3 .eq. 10 .and. k .eq. 5)
              end if
              if (iconfg .eq. 2) then
                 good = (i3 .eq. 1 .and. k .eq. 6)
     &              .or. (i3 .eq. 10 .and. k .eq. 5)
              end if
              if (good) then

c                j = icell
                 call newcre(i, icell)
c                 icell = j

                 icels(i) = 111111

c2.2    particle moves inside the cube
              else

                 if (k .eq. 1) i1 = i1 + 1
                 if (k .eq. 2) i1 = i1 - 1
                 if (k .eq. 3) i2 = i2 + 1
                 if (k .eq. 4) i2 = i2 - 1
                 if (k .eq. 5) i3 = i3 + 1
                 if (k .eq. 6) i3 = i3 - 1
                 
                 if (iconfg .eq. 2 .or. iconfg .eq. 4) then
                    if (i1 .eq. 0) then
                       i1 = 10
                       gx(i) = gx(i) + 10d0 * size1
                    end if
                    if (i1 .eq. 11) then
                       i1 = 1
                       gx(i) = gx(i) - 10d0 * size1
                    end if
                    if (i2 .eq. 0) then
                       i2 = 10
                       gy(i) = gy(i) + 10d0 * size2
                    end if
                    if (i2 .eq. 11) then
                       i2 = 1
                       gy(i) = gy(i) - 10d0 * size2
                    end if
                    if (iconfg .eq. 4) then
                       if (i3 .eq. 0) then
                          i3 = 10
                          gz(i) = gz(i) + 10d0 * size3
                       end if
                       if (i3 .eq. 11) then
                          i3 = 1
                          gz(i) = gz(i) - 10d0 * size3
                       end if
                    end if
                 end if
                 
                 j = icel(i1, i2, i3)
                 
                 call newcre(i, j)
c       in case icel changes
                 
                 icel(i1 ,i2, i3) = j
                 
                 icels(i) = i1 * 10000 + i2 * 100 + i3
                 
              end if
              
cc       for particles outside the cube
           else
              
              if (icell .eq. i) icell = nic(i)
              
              call oldcre(i)
              
              k = mod(icsta(i), 10)
              
              ddt = t - ft(i)
              dtt = t - size
              if (dtt .le. 0d0) then
                 i1 = integ((gx(i) + vx(i) * ddt) / size1) + 6
                 i2 = integ((gy(i) + vy(i) * ddt) / size2) + 6
                 i3 = integ((gz(i) + vz(i) * ddt) / size3) + 6
              else
                 i1 = integ((gx(i) + vx(i) * ddt) / 
     &               (size1 + v1 * dtt)) + 6
                 i2 = integ((gy(i) + vy(i) * ddt) /
     &               (size2 + v2 * dtt)) + 6
                 i3 = integ((gz(i) + vz(i) * ddt) /
     &               (size3 + v3 * dtt)) + 6
              end if 


              if (k .eq. 1) i1 = 1
              if (k .eq. 2) i1 = 10
              if (k .eq. 3) i2 = 1
              if (k .eq. 4) i2 = 10
              if (k .eq. 5) i3 = 1
              if (k .eq. 6) i3 = 10

              j = icel(i1, i2, i3)
              call newcre(i, j)
              icel(i1, i2, i3) = j
              
              icels(i) = i1 * 10000 + i2 * 100 + i3
              
           end if
        end if

        if (next(i) .ne. 0) then
           otmp = ot(next(i))
           ctmp = ct(next(i))
        end if

        if (i1 .eq. 11 .and. i2 .eq. 11 .and. i3 .eq. 11) then
           call dchout(i, k, t)
        else
           if (iconfg .eq. 1) then
              call dchin1(i, k, i1, i2, i3, t)
           else if (iconfg .eq. 2) then
              call dchin2(i, k, i1, i2, i3, t)
           else if (iconfg .eq. 4) then
              call dchin3(i, k, i1, i2, i3, t)              
           end if
        end if

        if (icsta(i) / 10 .eq. 11) then
           ot(next(i)) = otmp
           ct(next(i)) = ctmp
           next(next(i)) = i
           call wallc(i, i1, i2, i3, t0, tmin1)
           if (tmin1 .lt. ct(i)) then
              icsta(i) = icsta(i) + 10
              t0 = tmin1
              goto 1000
           end if
        end if

        return
        end
           
        subroutine oldcre(i) 
c       this subroutine is used to rearrange the old cell nic when a particle 
c       goes out of the cell

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        SAVE   

        if (nic(i) .eq. 0) return

        j = nic(i)

        if (nic(j) .eq. i) then
           nic(j) = 0
           return
        end if

        do 10 while (nic(j) .ne. i)
           j = nic(j)
 10        continue
        
        nic(j) = nic(i)

        return
        end


        subroutine newcre(i, k)
c       this subroutine is used to mk rearrange of the new cell a particle
c       enters,
c       input i
c       output nic(i)

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        SAVE   

        if (k .eq. 0) then
           k = i
           nic(i) = 0
        else if (nic(k) .eq. 0) then
           nic(k) = i
           nic(i) = k
        else
           j = k
           do 10 while (nic(j) .ne. k)
              j = nic(j)
 10           continue

           nic(j) = i
           nic(i) = k

        end if
        
        return
        end

        subroutine scat(t, iscat, jscat)

c       this subroutine is used to calculate the 2 particle scattering

        implicit double precision (a-h, o-z)
        SAVE   

        call newpos(t, iscat)
        call newpos(t, jscat)
        call newmom(t)

        return
        end

        subroutine newpos(t, i)

c       this subroutine is used to calculate the 2 particle scattering
c       get new position

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /prec5/ eta(MAXPTN), rap(MAXPTN), tau(MAXPTN)
cc      SAVE /prec5/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        t=t
        dt1 = ct(i) - ft(i)
        
        gx(i) = gx(i) + vx(i) * dt1
        gy(i) = gy(i) + vy(i) * dt1
        gz(i) = gz(i) + vz(i) * dt1
        ft(i) = ct(i)
           
        if (iconfg .le. 3) then
           if (ft(i) .le. abs(gz(i))) then
              eta(i) = 1000000.d0
           else
              eta(i) = 0.5d0 * log((ft(i) + gz(i)) / (ft(i) - gz(i)))
           end if
           tau(i) = ft(i) / cosh(eta(i))
        end if

        return
        end

        subroutine newmom(t)

c       this subroutine is used to calculate the 2 particle scattering

        implicit double precision (a-h, o-z)

        parameter (hbarc = 0.197327054d0)
        parameter (MAXPTN=400001)
        parameter (pi = 3.14159265358979d0)
        COMMON /para1/ mul
cc      SAVE /para1/
        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
ctrans
        common /para6/ centy
cc      SAVE /para6/
ctransend
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /prec5/ eta(MAXPTN), rap(MAXPTN), tau(MAXPTN)
cc      SAVE /prec5/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /lor/ enenew, pxnew, pynew, pznew
cc      SAVE /lor/
        common /cprod/ xn1, xn2, xn3
cc      SAVE /cprod/
        common /rndm2/ iff
cc      SAVE /rndm2/
        common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
        common /frzprc/ 
     &       gxfrz(MAXPTN), gyfrz(MAXPTN), gzfrz(MAXPTN), ftfrz(MAXPTN),
     &       pxfrz(MAXPTN), pyfrz(MAXPTN), pzfrz(MAXPTN), efrz(MAXPTN),
     &       xmfrz(MAXPTN), 
     &       tfrz(302), ifrz(MAXPTN), idfrz(MAXPTN), itlast
cc      SAVE /frzprc/
        SAVE   

        t=t
clin-6/06/02 no momentum change for partons already frozen out,
c     however, spatial upgrade is needed to ensure overall system freezeout:
      if(isoft.eq.5) then
         if(ifrz(iscat).eq.1.or.ifrz(jscat).eq.1) then
            last(iscat) = jscat
            last(jscat) = iscat
            return
         endif
      endif
clin-6/06/02-end

c       iff is used to randomize the interaction to have both attractive and
c        repulsive

        iff = - iff

        if (iconfg .eq. 2 .or. iconfg .eq. 4) then
           icels1 = icels(iscat)
           i1 = icels1 / 10000
           j1 = (icels1 - i1 * 10000) / 100
           icels2 = icels(jscat)
           i2 = icels2 / 10000
           j2 = (icels2 - i2 * 10000) / 100
           if (iconfg .eq. 4) then
              k1 = icels1 - i1 * 10000 - j1 * 100
              k2 = icels2 - i2 * 10000 - j2 * 100
           end if
        end if

        px1 = px(iscat)
        py1 = py(iscat)
        pz1 = pz(iscat)
        e1 = e(iscat)
        x1 = gx(iscat)
        y1 = gy(iscat)
        z1 = gz(iscat)
        t1 = ft(iscat)
        px2 = px(jscat)
        py2 = py(jscat)
        pz2 = pz(jscat)
        e2 = e(jscat)

        if (iconfg .eq. 1) then
           x2 = gx(jscat)
           y2 = gy(jscat)
           z2 = gz(jscat)
        else if (iconfg .eq. 2 .or. iconfg .eq. 4) then
           if (i1 - i2 .gt. 5) then
              x2 = gx(jscat) + 10d0 * size1
           else if (i1 - i2 .lt. -5) then
              x2 = gx(jscat) - 10d0 * size1
           else
              x2 = gx(jscat)
           end if
           if (j1 - j2 .gt. 5) then
              y2 = gy(jscat) + 10d0 * size2
           else if (j1 - j2 .lt. -5) then
              y2 = gy(jscat) - 10d0 * size2
           else
              y2 = gy(jscat)
           end if
           if (iconfg .eq. 4) then
              if (k1 - k2 .gt. 5) then
                 z2 = gz(jscat) + 10d0 * size3
              else if (k1 - k2 .lt. -5) then
                 z2 = gz(jscat) - 10d0 * size3
              else
                 z2 = gz(jscat)
              end if
           else
              z2 = gz(jscat)
           end if
        else if (iconfg .eq. 3 .or. iconfg .eq. 5) then
           x2 = gx(jscat) + dgxa(jscat)
           y2 = gy(jscat) + dgya(jscat)
           if (iconfg .eq. 5) then
              z2 = gz(jscat) + dgza(jscat)
           else
              z2 = gz(jscat)
           end if
        end if
        t2 = ft(jscat)
ctrans
        rts2 = (e1 + e2) ** 2 - (px1 + px2) ** 2 -
     &     (py1 + py2) ** 2 - (pz1 + pz2) ** 2
ctransend
        bex = (px1 + px2) / (e1 + e2)
        bey = (py1 + py2) / (e1 + e2)
        bez = (pz1 + pz2) / (e1 + e2)

        call lorenz(e1, px1, py1, pz1, bex, bey, bez)
cc      SAVE pxnew, ..., values for later use.
        px1 = pxnew
        py1 = pynew
        pz1 = pznew
        e1 = enenew

        pp2 = pxnew ** 2 + pynew ** 2 + pznew ** 2
        call getht(iscat, jscat, pp2, that)
        theta = dacos(that / (2d0 * pp2) + 1d0)
        theta = dble(iff) * theta

c       we boost to the cm frame, get rotation axis, and rotate 1 particle 
c       momentum

        call lorenz(t1, x1, y1, z1, bex, bey, bez)

        x1 = pxnew
        y1 = pynew
        z1 = pznew

        call lorenz(t2, x2, y2, z2, bex, bey, bez)

        x2 = pxnew
        y2 = pynew
        z2 = pznew

c       notice now pxnew, ..., are new positions
        call cropro(x1-x2, y1-y2, z1-z2, px1, py1, pz1)

        call xnormv(xn1, xn2, xn3)

cbz1/29/99
c        call rotate(xn1, xn2, xn3, theta, px1, py1, pz1)
        call zprota(xn1, xn2, xn3, theta, px1, py1, pz1)
cbz1/29/99end

c       we invert the momentum to get the other particle's momentum
        px2 = -px1
        py2 = -py1
        pz2 = -pz1
clin-4/13/01: modify in case m1, m2 are different:
c        e2 = e1
        e2 = dsqrt(px2**2+py2**2+pz2**2+xmass(jscat)**2)

c       boost the 2 particle 4 momentum back to lab frame
        call lorenz(e1, px1, py1, pz1, -bex, -bey, -bez)
        px(iscat) = pxnew
        py(iscat) = pynew
        pz(iscat) = pznew
        e(iscat) = enenew
        call lorenz(e2, px2, py2, pz2, -bex, -bey, -bez)        
        px(jscat) = pxnew
        py(jscat) = pynew
        pz(jscat) = pznew
        e(jscat) = enenew

        vx(iscat) = px(iscat) / e(iscat)
        vy(iscat) = py(iscat) / e(iscat)
        vz(iscat) = pz(iscat) / e(iscat)
        vx(jscat) = px(jscat) / e(jscat)
        vy(jscat) = py(jscat) / e(jscat)
        vz(jscat) = pz(jscat) / e(jscat)
        
        last(iscat) = jscat
        last(jscat) = iscat

        if (iconfg .le. 3) then
           if (e(iscat) .le. abs(pz(iscat))) then
              rap(iscat) = 1000000.d0
           else
              rap(iscat) = 0.5d0 * log((e(iscat) + pz(iscat)) /
     &           (e(iscat) - pz(iscat)))
           end if

           if (e(jscat) .le. abs(pz(jscat))) then
              rap(jscat) = 1000000.d0
           else
              rap(jscat) = 0.5d0 * log((e(jscat) + pz(jscat)) /
     &           (e(jscat) - pz(jscat)))
           end if

ctrans
           rap1 = rap(iscat)
           rap2 = rap(jscat)

           if ((rap1 .lt. centy + 0.5d0 .and.
     &        rap1 .gt. centy - 0.5d0)) then
c              write (9, *) sqrt(ft(iscat) ** 2 - gz(iscat) ** 2), rts2
           end if
           if ((rap2 .lt. centy + 0.5d0 .and.
     &        rap2 .gt. centy - 0.5d0)) then
c              write (9, *) sqrt(ft(jscat) ** 2 - gz(jscat) ** 2), rts2
           end if
ctransend
        end if

        return
        end

        subroutine getht(iscat, jscat, pp2, that)

c       this subroutine is used to get \hat{t} for a particular processes

        implicit double precision (a-h, o-z)

        parameter (hbarc = 0.197327054d0)
        parameter (MAXPTN=400001)
        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
cc        external ran1
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        SAVE   

        iscat=iscat
        jscat=jscat
        iseed=iseedp
        xmu2 = (hbarc * xmu) ** 2
        xmp2 = xmp ** 2
        xm2 = xmu2 + xmp2
        rx=ran1(iseed)
        that = xm2*(1d0+1d0/((1d0-xm2/(4d0*pp2+xm2))*rx-1d0))
ctest off isotropic scattering:
c     &     + 1d0/((1d0 - xm2 / (4d0 * pp2 + xm2)) * ran1(2) - 1d0))
c        if(izpc.eq.100) that=-4d0*pp2*ran1(2)
        if(izpc.eq.100) that=-4d0*pp2*rx

        return
        end

      subroutine ulist(t)
c     this subroutine is used to update a new collision time list
c       notice this t has been updated

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        SAVE   

        if (ictype .eq. 1 .or. ictype .eq. 2 .or. ictype .eq. 5
     &     .or. ictype .eq. 6) then
           l = ifmpt
           call ulist1(l, t)
        end if
        if (ictype .ne. 1) then
           l = iscat
           call ulist1(l, t)
           if (jscat .ne. 0) then
              l = jscat
              call ulist1(l, t)
           end if
        end if

        return
        end

        subroutine ulist1(l, t)
c       this subroutine is used to update the interaction list when particle
c       l is disturbed.

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        icels0 = icels(l)
        i1 = icels0 / 10000
        i2 = (icels0 - i1 * 10000) / 100
        i3 = icels0 - i1 * 10000 - i2 * 100
c       save collision info for use when the collision is a collision with wall
c       otherwise wallc will change icsta
        k = mod(icsta(l), 10)

        call wallc(l, i1, i2, i3, t, tmin1)
        tmin = tmin1
        nc = 0

        if (i1 .eq. 11 .and. i2 .eq. 11 .and. i3 .eq. 11) then
           call chkout(l, t, tmin, nc)
        else
           if (iconfg .eq. 1) then
              call chkin1(l, i1, i2, i3, t, tmin, nc)
           else if (iconfg .eq. 2) then
              call chkin2(l, i1, i2, i3, t, tmin, nc)
           else if (iconfg .eq. 4) then
              call chkin3(l, i1, i2, i3, t, tmin, nc)
           else if (iconfg .eq. 3 .or. iconfg .eq. 5) then
              call chkcel(l, i1, i2, i3, t, tmin, nc)
           end if
        end if
        
        call fixtim(l, t, tmin1, tmin, nc)

        return
        end
        
        subroutine wallc(i, i1, i2, i3, t, tmin)
c       this subroutine calculates the next time for collision with wall 
c       for particle i
c       input particle label i,t
c       output tmin collision time with wall, icsta(i) wall collision
c       information

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        tmin = tlarge

        if (iconfg .le. 2 .or. iconfg .eq. 4) then
c       if particle is inside the cube
           if ((i1 .ge. 1 .and. i1 .le. 10)
     &          .or. (i2 .ge. 1 .and. i2 .le. 10)
     &          .or. (i3 .ge. 1 .and. i3 .le. 10)) then
              call wallc1(i, i1, i2, i3, t, tmin)
c       if particle is outside the cube
           else
              call wallcb(i, t, tmin)              
           end if
        else if (iconfg .eq. 3 .or. iconfg .eq. 5) then
           call wallc2(i, i1, i2, i3, t, tmin)
        end if

        return
        end

        subroutine wallc1(i, i1, i2, i3, t, tmin)
c       this subroutine is used to get wall collision time
c       when particle is inside the cube, it sets the icsta at the same time
c       input i,i1,i2,i3,t
c       output tmin, icsta(i)
c       note the icsta is not finally set. we need further judgement in 
c       fixtim

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        x1p = gx(i)
        x2p = gy(i)
        x3p = gz(i)
        tf = ft(i)
        v1p = vx(i)
        v2p = vy(i)
        v3p = vz(i)

        if (t .lt. size .and. tf .lt. size) then

           if (v1p .gt. 0d0) then
              t1 = ((dble(i1) - 5d0) * size1 - x1p) / v1p + tf
           else if (v1p .lt. 0d0) then
              t1 = ((dble(i1) - 6d0) * size1 - x1p) / v1p + tf
           else
              t1 = tlarge
           end if
           
           if (v2p .gt. 0d0) then
              t2 = ((dble(i2) - 5d0) * size2 - x2p) / v2p + tf
           else if (v2p .lt. 0d0) then
              t2 = ((dble(i2) - 6d0) * size2 - x2p) / v2p + tf
           else
              t2 = tlarge
           end if
           
           if (v3p .gt. 0d0) then
              t3 = ((dble(i3) - 5d0) * size3 - x3p) / v3p + tf
           else if (v3p .lt. 0d0) then
              t3 = ((dble(i3) - 6d0) * size3 - x3p) / v3p + tf
           else
              t3 = tlarge
           end if
           
c       if a particle is on the wall, we don't collide it on the same wall
           
c        if (t1 .eq. 0d0) t1 = tlarge
c        if (t2 .eq. 0d0) t2 = tlarge
c        if (t3 .eq. 0d0) t3 = tlarge
           
           tmin = min(t1, t2, t3)
           
c       set icsta,
c       after checking this is not an earlier collision comparing with 
c       a collision with another particle, we need to set icsta=0
c       after checking whether there is also a particle collision 
c       at the same time, we need to reset the second bit of icsta
           
           if (tmin .eq. t1) then
              if (v1p .gt. 0d0) then
                 icsta(i) = 101
              else
                 icsta(i) = 102
              end if
           end if
           
           if (tmin .eq. t2) then
              if (v2p .gt. 0d0) then
                 icsta(i) = 103
              else
                 icsta(i) = 104
              end if
           end if
           
           if (tmin .eq. t3) then
              if (v3p .gt. 0d0) then
                 icsta(i) = 105
              else
                 icsta(i) = 106
              end if
           end if
           
        if (tmin .le. size) return

        end if

        if (v1p .gt. (i1 - 5) * v1) then
           t1 = ((i1 - 5) * (size1 - v1 * size) +
     &          v1p * tf - x1p) / (v1p - (i1 - 5) * v1)
        else if (v1p .lt. (i1 - 6) * v1) then
           t1 = ((i1 - 6) * (size1 - v1 * size) +
     &          v1p * tf - x1p) / (v1p - (i1 - 6) * v1)
        else
           t1 = tlarge
        end if
        
        if (v2p .gt. (i2 - 5) * v2) then
           t2 = ((i2 - 5) * (size2 - v2 * size) +
     &          v2p * tf - x2p) / (v2p - (i2 - 5) * v2)
        else if (v2p .lt. (i2 - 6) * v2) then
           t2 = ((i2 - 6) * (size2 - v2 * size) +
     &          v2p * tf - x2p) / (v2p - (i2 - 6) * v2)
        else
           t2 = tlarge
        end if
        
        if (v3p .gt. (i3 - 5) * v3) then
           t3 = ((i3 - 5) * (size3 - v3 * size) +
     &          v3p * tf - x3p) / (v3p - (i3 - 5) * v3)
        else if (v3p .lt. (i3 - 6) * v3) then
           t3 = ((i3 - 6) * (size3 - v3 * size) +
     &          v3p * tf - x3p) / (v3p - (i3 - 6) * v3)
        else
           t3 = tlarge
        end if
        
c       if a particle is on the wall, we don't collide it on the same wall
        
c        if (t1 .eq. 0d0) t1 = tlarge
c        if (t2 .eq. 0d0) t2 = tlarge
c        if (t3 .eq. 0d0) t3 = tlarge
        
        tmin = min(t1, t2, t3)
        
c       set icsta,
c       after checking this is not an earlier collision comparing with 
c       a collision with another particle, we need to set icsta=0
c       after checking whether there is also a particle collision 
c       at the same time, we need to reset the second bit of icsta
        
        if (tmin .eq. t1) then
           if (v1p .gt. (i1 - 5) * v1) then
              icsta(i) = 101
           else
              icsta(i) = 102
           end if
        end if
        
        if (tmin .eq. t2) then
           if (v2p .gt. (i2 - 5) * v2) then
              icsta(i) = 103
           else
              icsta(i) = 104
           end if
        end if
        
        if (tmin .eq. t3) then
           if (v3p .gt. (i3 - 5) * v3) then
              icsta(i) = 105
           else
              icsta(i) = 106
           end if
        end if
        
        return
        end

        subroutine wallc2(i, i1, i2, i3, t, tmin)
c       this subroutine is used to get wall collision time
c       when particle is inside the cube, it sets the icsta at the same time
c       input i,i1,i2,i3,t
c       output tmin, icsta(i)
c       note the icsta is not finally set. we need further judgement in 
c       fixtim

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        i1=i1
        i2=i2
        i3=i3
        t=t
        x1p = gx(i)
        x2p = gy(i)
        x3p = gz(i)
        tf = ft(i)
        v1p = vx(i)
        v2p = vy(i)
        v3p = vz(i)

        if (v1p .gt. 0d0) then
           t1 = (5d0 * size1 - x1p) / v1p + tf
        else if (v1p .lt. 0d0) then
           t1 = (-5d0 * size1 - x1p) / v1p + tf
        else
           t1 = tlarge
        end if
        
        if (v2p .gt. 0d0) then
           t2 = (5d0 * size2 - x2p) / v2p + tf
        else if (v2p .lt. 0d0) then
           t2 = (- 5d0 * size2 - x2p) / v2p +tf
        else
           t2 = tlarge
        end if

        if (iconfg .eq. 5) then
           if (v3p .gt. 0d0) then
              t3 = (5d0 * size3 - x3p) / v3p + tf
           else if (v3p .lt. 0d0) then
              t3 = (- 5d0 * size3 - x3p) / v3p +tf
           else
              t3 = tlarge
           end if
        else
           t3 = tlarge
        end if
           
        tmin = min(t1, t2, t3)
        
c       set icsta,
c       after checking this is not an earlier collision comparing with 
c       a collision with another particle, we need to set icsta=0
c       after checking whether there is also a particle collision 
c       at the same time, we need to reset the second bit of icsta
           
        if (tmin .eq. t1) then
           if (v1p .gt. 0d0) then
              icsta(i) = 101
           else
              icsta(i) = 102
           end if
        end if
        
        if (tmin .eq. t2) then
           if (v2p .gt. 0d0) then
              icsta(i) = 103
           else
              icsta(i) = 104
           end if
        end if

        if (tmin .eq. t3) then
           if (v3p .gt. 0d0) then
              icsta(i) = 105
           else
              icsta(i) = 106
           end if
        end if
           
        return
        end

        subroutine wallcb(i, t, tmin)
c       this subroutine is used to calculate the wall collision time 
c       when the particle is outside the cube
c       input i,t
c       output tmin,icsta(i)
c       note the icsta is not finally set. we need further judgement in 
c       fixtim

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

c       check if there is a collision by looking at the closest approach point
c       and see if it's inside the cube

        if (size1 .eq. 0d0 .or. size2 .eq. 0d0 .or. 
     &     size3 .eq. 0d0) return

        x1p = gx(i)
        x2p = gy(i)
        x3p = gz(i)
        v1p = vx(i)
        v2p = vy(i)
        v3p = vz(i)
        tf = ft(i)

        if (t .lt. size .and. tf .lt. size) then
           if (x1p .lt. - 5d0 * size1 .and. v1p .gt. 0d0) then
              t1 = (- 5d0 * size1 - x1p) / v1p + tf
           else if(x1p .gt. 5d0 * size1 .and. v1p .lt. 0d0) then
              t1 = - (x1p - 5d0 * size1) / v1p + tf
           else
              t1 = tlarge 
           end if

           if (t1 .ne. tlarge) then
              x2pp = x2p + v2p * (t1 - tf)
              x3pp = x3p + v3p * (t1 - tf)
              if (x2pp .le. - 5d0 * size2 .or. x2pp .ge. 5d0 * size2
     &             .or. x3pp .le. - 5d0 * size3 
     &             .or. x3pp .ge. 5d0 * size3)
     &             t1 = tlarge
           end if
           
           if (x2p .lt. - 5d0 * size2 .and. v2p .gt. 0d0) then
              t2 = (- 5d0 * size2 - x2p) / v2p + tf
           else if(x2p .gt. 5d0 * size2 .and. v2p .lt. 0d0) then
              t2 = - (x2p - 5d0 * size2) / v2p + tf
           else
              t2 = tlarge 
           end if
           
           if (t2 .ne. tlarge) then
              x1pp = x1p + v1p * (t2 - tf)
              x3pp = x3p + v3p * (t2 - tf)
              if (x1pp .le. - 5d0 * size1 .or. x1pp .ge. 5d0 * size1
     &          .or. x3pp .le. - 5d0 * size3 .or. x3pp .ge. 5d0 * size3)
     &             t2 = tlarge
           end if
           
           if (x3p .lt. - 5d0 * size3 .and. v3p .gt. 0d0) then
              t3 = (- 5d0 * size3 - x3p) / v3p + tf
           else if(x3p .gt. 5d0 * size3 .and. v3p .lt. 0d0) then
              t3 = - (x3p - 5d0 * size3) / v3p + tf
           else
              t3 = tlarge 
           end if
           
           if (t3 .ne. tlarge) then
              x1pp = x1p + v1p * (t3 - tf)
              x2pp = x2p + v2p * (t3 - tf)
              if (x1pp .le. - 5d0 * size1 .or. x1pp .ge. 5d0 * size1
     &          .or. x2pp .le. - 5d0 * size2 .or. x2pp .ge. 5d0 * size2)
     &             t3 = tlarge
           end if
           
           tmin = min(t1, t2, t3)

c       set icsta,
c       after checking this is not an earlier collision comparing with 
c       a collision with another particle, we need to set icsta=0
c       after checking whether there is also a particle collision 
c       at the same time, we need to reset the second bit of icsta

           if (tmin .eq. t1) then
              if (v1p .gt. 0d0) then
                 icsta(i) = 101
              else
                 icsta(i) = 102
              end if
           end if
           
           if (tmin .eq. t2) then
              if (v2p .gt. 0d0) then
                 icsta(i) = 103
              else
                 icsta(i) = 104
              end if
           end if
        
           if (tmin .eq. t3) then
              if (v3p .gt. 0d0) then
                 icsta(i) = 105
              else
                 icsta(i) = 106
              end if
           end if
           
        if (tmin .le. size) return

        end if

c       notice now x1q, x2q, x3q are coordinates at time t
        x1q = x1p + v1p * (t - tf)
        x2q = x2p + v2p * (t - tf)
        x3q = x3p + v3p * (t - tf)

        if (x1q .lt. - 5d0 * (size1 + v1 * (t - size)) .and. 
     &      v1p .gt. - 5d0 * v1) then
           t1 = (- 5d0 * (size1 - v1 * size) + v1p * tf - x1p) /
     &          (v1p - (- 5d0) * v1)
           icsta1 = 101
        else if (x1q .gt. 5d0 * (size1 + v1 * (t-size)) .and. 
     &     v1p .lt. 5d0 * v1) then
           t1 = (5d0 * (size1 - v1 * size) + v1p * tf - x1p) /
     &          (v1p - 5d0 * v1)
           icsta1 = 102
        else
           t1 = tlarge 
        end if
        
        if (t1 .ne. tlarge) then
           x2pp = x2p + v2p * (t1 - tf)
           x3pp = x3p + v3p * (t1 - tf)
           if (x2pp .le. - 5d0 * (size2 + v2 * (t1 - size))
     &        .or. x2pp .ge. 5d0 * (size2 + v2 * (t1 - size))
     &        .or. x3pp .le. - 5d0 * (size3 + v3 * (t1 - size))
     &        .or. x3pp .ge. 5d0 * (size3 + v3 * (t1 - size)))
     &        t1 = tlarge
        end if

        if (x2q .lt. - 5d0 * (size2 + v2 * (t - size)) .and.
     &     v2p .gt. - 5d0 * v2) then
           t2 = (- 5d0 * (size2 - v2 * size) + v2p * tf - x2p) /
     &          (v2p - (- 5d0) * v2)
           icsta2 = 103
        else if (x2q .gt. 5d0 * (size2 + v2 * (t - size)) .and.
     &     v2p .lt. 5d0 * v2) then
           t2 = (5d0 * (size2 - v2 * size) + v2p * tf - x2p) / 
     &          (v2p - 5d0 * v2)
           icsta2 = 104
        else
           t2 = tlarge 
        end if
        
        if (t2 .ne. tlarge) then
           x1pp = x1p + v1p * (t2 - tf)
           x3pp = x3p + v3p * (t2 - tf)
           if (x1pp .le. - 5d0 * (size1 + v1 * (t2 - size))
     &        .or. x1pp .ge. 5d0 * (size1 + v1 * (t2 - size))
     &        .or. x3pp .le. - 5d0 * (size3 + v3 * (t2 - size))
     &        .or. x3pp .ge. 5d0 * (size3 + v3 * (t2 - size)))
     &        t2 = tlarge
        end if

        if (x3q .lt. - 5d0 * (size3 + v3 * (t - size)) .and. 
     &     v3p .gt. - 5d0 * v3) then
           t3 = (- 5d0 * (size3 - v3 * size) + v3p * tf - x3p) /
     &          (v3p - (- 5d0) * v3)
           icsta3 = 105
        else if (x3q .gt. 5d0 * (size3 + v3 * (t - size)) .and.
     &     v3p .lt. 5d0 * v3) then
           t3 = (5d0 * (size3 - v3 * size) + v3p * tf - x3p) /
     &          (v3p - 5d0 * v3)
           icsta3 = 106
        else
           t3 = tlarge 
        end if
        
        if (t3 .ne. tlarge) then
           x2pp = x2p + v2p * (t3 - tf)
           x1pp = x1p + v1p * (t3 - tf)
           if (x2pp .le. - 5d0 * (size2 + v2 * (t3 - size))
     &        .or. x2pp .ge. 5d0 * (size2 + v2 * (t3 - size))
     &        .or. x1pp .le. - 5d0 * (size1 + v1 * (t3 - size))
     &        .or. x1pp .ge. 5d0 * (size1 + v1 * (t3 - size)))
     &        t3 = tlarge
        end if
        
        tmin = min(t1, t2, t3)
        
c       set icsta,
c       after checking this is not an earlier collision comparing with 
c       a collision with another particle, we need to set icsta=0
c       after checking whether there is also a particle collision 
c       at the same time, we need to reset the second bit of icsta
        
        if (tmin .eq. t1) then
           icsta(i) = icsta1
        else if (tmin .eq. t2) then
           icsta(i) = icsta2
        else if (tmin .eq. t3) then
           icsta(i) = icsta3
        end if
        
        return
        end
           
        subroutine chkout(l, t, tmin, nc)
c       this subroutine is used to check the collisions with particles in 
c       surface cells to see if we can get a smaller collision time than tmin
c       with particle nc, when the colliding particle is outside the cube
c       input l,t,tmin,nc
c       output tmin, nc

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        SAVE   

        m1 = 11
        m2 = 11
        m3 = 11
        call chkcel(l, m1, m2, m3, t, tmin, nc)

        do 1003 i = 1, 10
           do 1002 j = 1, 10
              do 1001 k = 1, 10
                 if (i .eq. 1 .or. i .eq. 10 .or. j .eq. 1
     &              .or. j .eq. 10 .or. k .eq. 1 .or. k .eq. 10) 
     &                    call chkcel(l, i, j, k, t, tmin, nc)
 1001         continue
 1002      continue
 1003   continue

        return
        end

        subroutine chkin1(l, i1, i2, i3, t, tmin, nc)
c       this subroutine is used to check collisions for particle inside
c       the cube
c       and update the afftected particles through chkcel

        implicit double precision (a-h, o-z)
        SAVE   

c       itest is a flag to make sure the 111111 cell is checked only once
        itest = 0
        
        do 1003 i = i1 - 1, i1 + 1
           do 1002 j = i2 - 1, i2 + 1
              do 1001 k =  i3 - 1, i3 + 1
                 if (i .ge. 1 .and. i .le. 10 .and. j .ge. 1 .and.
     &               j .le. 10 .and. k .ge. 1 .and. k .le. 10) then
                    call chkcel(l, i, j, k, t, tmin, nc)
                 else if (itest .eq. 0) then
                    m1 = 11
                    m2 = 11
                    m3 = 11
                    call chkcel(l, m1, m2, m3, t, tmin, nc)
                    itest = 1
                 end if   
 1001         continue
 1002      continue
 1003   continue

        return
        end

        subroutine chkin2(l, i1, i2, i3, t, tmin, nc)
c       this subroutine is used to check collisions for particle inside
c       the cube
c       and update the afftected particles through chkcel

        implicit double precision (a-h, o-z)
        SAVE   

c       itest is a flag to make sure the 111111 cell is checked only once
        itest = 0
        
        do 1003 i = i1 - 1, i1 + 1
           do 1002 j = i2 - 1, i2 + 1
              do 1001 k =  i3 - 1, i3 + 1
                 ia = i
                 ib = j
                 ic = k
                 if (k .ge. 1 .and. k .le. 10) then
                    if (i .eq. 0) ia = 10
                    if (i .eq. 11) ia = 1
                    if (j .eq. 0) ib = 10
                    if (j .eq. 11) ib = 1
                    call chkcel(l, ia, ib, ic, t, tmin, nc)
                 end if
 1001         continue
 1002      continue
 1003   continue

        return
        end

        subroutine chkin3(l, i1, i2, i3, t, tmin, nc)
c       this subroutine is used to check collisions for particle inside
c       the cube
c       and update the afftected particles through chkcel

        implicit double precision (a-h, o-z)
        SAVE   

c       itest is a flag to make sure the 111111 cell is checked only once
        itest = 0
        
        do 1003 i = i1 - 1, i1 + 1
           do 1002 j = i2 - 1, i2 + 1
              do 1001 k =  i3 - 1, i3 + 1
                 if (i .eq. 0) then
                    ia = 10
                 else if (i .eq. 11) then
                    ia = 1
                 else
                    ia = i
                 end if
                 if (j .eq. 0) then
                    ib = 10
                 else if (j .eq. 11) then
                    ib = 1
                 else
                    ib = j
                 end if
                 if (k .eq. 0) then
                    ic = 10
                 else if (k .eq. 11) then
                    ic = 1
                 else
                    ic = k
                 end if
                 call chkcel(l, ia, ib, ic, t, tmin, nc)
 1001         continue
 1002      continue
 1003   continue

        return
        end

        subroutine chkcel(il, i1, i2, i3, t, tmin, nc)
c       this program is used to check through all the particles
c       in the cell (i1,i2,i3) and see if we can get a particle collision 
c       with time less than the original input tmin ( the collision time of 
c       il with the wall
c       and update the affected particles

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist2/ icell, icel(10, 10, 10)
cc      SAVE /ilist2/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        SAVE   

        if (iconfg .eq. 3 .or. iconfg .eq. 5) then
           jj = ichkpt
           do 1001 j = 1, jj
              call ck(j, ick)
c     10/24/02 get rid of argument usage mismatch in ud2():
                            jud2=j
c              if (ick .eq. 1) call ud2(j, il, t, tmin, nc)
              if (ick .eq. 1) call ud2(jud2, il, t, tmin, nc)
 1001      continue
           return
        end if

        if (i1 .eq. 11 .and. i2 .eq. 11 .and. i3 .eq. 11) then
           l = icell
        else
           l = icel(i1, i2, i3)
        end if

c       if there is no particle
        if (l .eq. 0) then
           return
        end if
        j = nic(l)
c       if there is only one particle
        if (j .eq. 0) then
           call ck(l, ick)
           if (ick .eq. 1) call ud2(l, il, t, tmin, nc)

c       if there are many particles
        else

c       we don't worry about the other colliding particle because it's
c       set in last(), and will be checked in ud2

           call ck(l, ick)
           if (ick .eq. 1) call ud2(l, il, t, tmin, nc)

           do 10 while(j .ne. l)
              call ck(j, ick)
              if (ick .eq. 1) call ud2(j, il, t, tmin, nc)
              j = nic(j)
 10           continue
        end if

        return
        end

        subroutine ck(l, ick)
c       this subroutine is used for chcell to check whether l should be
c       checked or not for updating tmin, nc
c       input l
c       output ick
c       if ick=1, l should be checked, otherwise it should not be.

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        SAVE   

        ick = 1
        if (ictype .eq. 1) then
           if (l .eq. ifmpt) ick = 0
        else if (ictype .eq. 0 .or. ictype .eq. 3 .or. 
     &     ictype .eq. 4) then
           if (l .eq. iscat .or. l .eq. jscat) ick = 0
        else
           if (l .eq. iscat .or. l .eq. jscat .or.
     &         l .eq. ifmpt) ick = 0
        end if
c       notice il is either iscat or jscat, or ifmpt, we deal with them
c       seperately according to ictype

        return
        end
           
        subroutine dchout(l, ii, t)
c       this subroutine is used to check collisions of l with particles when 
c       l is outside the cube and the collision just happened is a collision
c       including a collision with wall (hence we need to use dcheck to throw
c       away old collisions that are not in the new neighboring cells.

c       input l,t

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        external integ

        tt = ft(l)
        td = t - size
        x1 = gx(l) + vx(l) * (t - tt)
        x2 = gy(l) + vy(l) * (t - tt)
        x3 = gz(l) + vz(l) * (t - tt)
        if (td .le. 0d0) then
           i1 = integ(x1 / size1) + 6
           i2 = integ(x2 / size2 ) + 6
           i3 = integ(x3 / size3 ) + 6
           if (integ(x1 / size1) .eq. x1 / size1 .and. vx(l) .lt. 0d0)
     &        i1 = i1 - 1
           if (integ(x2 / size2) .eq. x2 / size2 .and. vy(l) .lt. 0d0)
     &        i2 = i2 - 1
           if (integ(x3 / size3) .eq. x3 / size3 .and. vz(l) .lt. 0d0)
     &        i3 = i3 - 1
        else
           i1 = integ(x1 / (size1 + v1 * td)) + 6
           i2 = integ(x2 / (size2 + v2 * td)) + 6
           i3 = integ(x3 / (size3 + v3 * td)) + 6
c     10/24/02 (i) below should be (l):
           if (integ(x1 / (size1 + v1 * td)) .eq. 
     &        x1 / (size1 +v1 * td) .and. 
     &        vx(l) .lt. (i1 - 6) * v1) i1 = i1 - 1
c     &        vx(i) .lt. (i1 - 6) * v1) i1 = i1 - 1
           if (integ(x2 / (size2 + v2 * td)) .eq.
     &        x2 / (size2 + v2 * td) .and.
     &        vy(l) .lt. (i2 - 6) * v2) i2 = i2 - 1
c     &        vy(i) .lt. (i2 - 6) * v2) i2 = i2 - 1
           if (integ(x3 / (size3 + v3 * td)) .eq. 
     &        x3 / (size3 + v3 * td) .and.
     &        vz(l) .lt. (i3 - 6) * v3) i3 = i3 - 1
c     &        vz(i) .lt. (i3 - 6) * v3) i3 = i3 - 1
        end if

        if (ii .eq. 1) then
           i = 9
           do 1002 j = i2 - 1, i2 + 1
              do 1001 k = i3 - 1, i3 + 1
                 if (j .ge. 1 .and. j .le. 10 .and. k .ge. 1 .and.
     &              k .le. 10) then
                    call dchcel(l, i, j, k, t)
                 end if
 1001         continue
 1002      continue
        end if

        if (ii .eq. 2) then
           i = 2
           do 1004 j = i2 - 1, i2 + 1
              do 1003 k = i3 - 1, i3 + 1
                 if (j .ge. 1 .and. j .le. 10 .and. k .ge. 1 .and. 
     &              k .le. 10) then
                    call dchcel(l, i, j, k, t)
                 end if
 1003         continue
 1004      continue
        end if

        if (ii .eq. 3) then
           j = 9
           do 1006 i = i1 - 1, i1 + 1
              do 1005 k = i3 - 1, i3 + 1
                 if (i .ge. 1 .and. i .le. 10 .and. k .ge. 1 .and.
     &              k .le. 10) then
                    call dchcel(l, i, j, k, t)
                 end if
 1005         continue
 1006      continue
        end if

        if (ii .eq. 4) then
           j = 2
           do 1008 i = i1 - 1, i1 + 1
              do 1007 k = i3 - 1, i3 + 1
                 if (i .ge. 1 .and. i .le. 10 .and. k .ge. 1 .and.
     &              k .le. 10) then
                    call dchcel(l, i, j, k, t)
                 end if
 1007         continue
 1008      continue
        end if

        if (ii .eq. 5) then
           k = 9
           do 1010 i = i1 - 1, i1 + 1
              do 1009 j = i2 - 1, i2 + 1
                 if (i .ge. 1 .and. i .le. 10 .and. j .ge. 1 .and.
     &              j .le. 10) then
                    call dchcel(l, i, j, k, t)
                 end if
 1009         continue
 1010      continue
        end if

        if (ii .eq. 6) then
           k = 2
           do 1012 i = i1 - 1, i1 + 1
              do 1011 j = i2 - 1, i2 + 1
                 if (i .ge. 1 .and. i .le. 10 .and. j .ge. 1 .and.
     &              j .le. 10) then
                    call dchcel(l, i, j, k, t)
                 end if
 1011         continue
 1012      continue
        end if

        return
        end

        subroutine dchin1(l, ii, i1, i2, i3, t)
c       this subroutine is used to check collisions for particle inside
c       the cube when the collision just happened is a collision including 
c       collision with wall
c       and update the afftected particles through chkcel

c       input l,ii(specifying the direction of the wall collision),
c          i1,i2,i3, (specifying the position of the cell 
c                    we are going to check)
c          t

        implicit double precision (a-h, o-z)
        SAVE   

c       itest is a flag to make sure the 111111 cell is checked only once
        itest = 0
        
        if (ii .eq. 1) then
           if (i1 .eq. 1) goto 100
           if (i1 .eq. 2) then
              if (i2 .ge. 2 .and. i2 .le. 9 .and. i3 .ge. 2 .and.
     &           i3 .le. 9) then
                 i = 11
                 j = 11
                 k = 11
                 call dchcel(l, i, j, k, t)
              end if
              goto 100
           end if
           i = i1 - 2
           do 1002 j = i2 - 1, i2 + 1
              do 1001 k = i3 - 1, i3 + 1
                 if (j .ge. 1 .and. j .le. 10 .and. k .ge. 1 .and.
     &              k .le. 10)
     &                    call dchcel(l, i, j, k, t)
 1001         continue
 1002      continue
        end if

        if (ii .eq. 2) then
           if (i1 .eq. 10) goto 100
           if (i1 .eq. 9) then
              if (i2 .ge. 2 .and. i2 .le. 9 .and. i3 .ge. 2 .and.
     &           i3 .le. 9) then
                 i = 11
                 j = 11
                 k = 11
                 call dchcel(l, i, j, k, t)
              end if
              goto 100
           end if
           i = i1 + 2
           do 1004 j = i2 - 1, i2 + 1
              do 1003 k = i3 - 1, i3 + 1
                 if (j .ge. 1 .and. j .le. 10 .and. k .ge. 1 .and.
     &              k .le. 10)
     &                    call dchcel(l, i, j, k, t)
 1003         continue
 1004      continue
        end if

        if (ii .eq. 3) then
           if (i2 .eq. 1) goto 100
           if (i2 .eq. 2) then
              if (i1 .ge. 2 .and. i1 .le. 9 .and. i3 .ge. 2 .and.
     &           i3 .le. 9) then
                 i = 11
                 j = 11
                 k = 11
                 call dchcel(l, i, j, k, t)
              end if
              goto 100
           end if
           j = i2 - 2
           do 1006 i = i1 - 1, i1 + 1
              do 1005 k = i3 - 1, i3 + 1
                 if (i .ge. 1 .and. i .le. 10 .and. k .ge. 1 .and.
     &              k .le. 10)
     &              call dchcel(l, i, j, k, t)
 1005         continue
 1006      continue
        end if

        if (ii .eq. 4) then
           if (i2 .eq. 10) goto 100
           if (i2 .eq. 9) then
              if (i1 .ge. 2 .and. i1 .le. 9 .and. i3 .ge. 2 .and.
     &           i3 .le. 9) then
                 i = 11
                 j = 11
                 k = 11
                 call dchcel(l, i, j, k, t)
              end if
              goto 100
           end if
           j = i2 + 2
           do 1008 i = i1 - 1, i1 + 1
              do 1007 k = i3 - 1, i3 + 1
                 if (i .ge. 1 .and. i .le. 10 .and. k .ge. 1 .and.
     &           k .le. 10)
     &                 call dchcel(l, i, j, k, t)
 1007         continue
 1008      continue
        end if

        if (ii .eq. 5) then
           if (i3 .eq. 1) goto 100
           if (i3 .eq. 2) then
              if (i1 .ge. 2 .and. i1 .le. 9 .and. i2 .ge. 2 .and.
     &           i2 .le. 9) then
                 i = 11
                 j = 11
                 k = 11
                 call dchcel(l, i, j, k, t)
              end if
              goto 100
           end if
           k = i3 - 2
           do 1010 i = i1 - 1, i1 + 1
              do 1009 j = i2 - 1, i2 + 1
                 if (i .ge. 1 .and. i .le. 10 .and. j .ge. 1 .and.
     &           j .le. 10)
     &                 call dchcel(l, i, j, k, t)
 1009         continue
 1010      continue
        end if

        if (ii .eq. 6) then
           if (i3 .eq. 10) goto 100
           if (i3 .eq. 9) then
              if (i1 .ge. 2 .and. i1 .le. 9 .and. i2 .ge. 2 .and.
     &           i2 .le. 9) then
                 i = 11
                 j = 11
                 k = 11
                 call dchcel(l, i, j, k, t)
              end if
              goto 100
           end if
           k = i3 + 2
           do 1012 i = i1 - 1, i1 + 1
              do 1011 j = i2 - 1, i2 + 1
                 if (i .ge. 1 .and. i .le. 10 .and. j .ge. 1 .and.
     &           j .le. 10)
     &                 call dchcel(l, i, j, k, t)
 1011         continue
 1012      continue
        end if

 100        continue

        return
        end

        subroutine dchin2(l, ii, i1, i2, i3, t)
c       this subroutine is used to check collisions for particle inside
c       the cube when the collision just happened is a collision including 
c       collision with wall
c       and update the afftected particles through chkcel

c       input l,ii(specifying the direction of the wall collision),
c          i1,i2,i3, (specifying the position of the cell 
c                    we are going to check)
c          t

        implicit double precision (a-h, o-z)
        SAVE   

        if (ii .eq. 1) then
           i = i1 - 2
           if (i .le. 0) i = i + 10
           ia = i
           do 1002 j = i2 - 1, i2 + 1
              do 1001 k = i3 - 1, i3 + 1
                 ib = j
                 ic = k
                 if (j .eq. 0) ib = 10
                 if (j .eq. 11) ib = 1
                 if (k .ge. 1 .and. k .le. 10) then
                    call dchcel(l, ia, ib, ic, t)
                 end if
 1001         continue
 1002      continue
        end if

        if (ii .eq. 2) then
           i = i1 + 2
           if (i .ge. 11) i = i - 10
           ia = i
           do 1004 j = i2 - 1, i2 + 1
              do 1003 k = i3 - 1, i3 + 1
                 ib = j
                 ic = k
                 if (j .eq. 0) ib = 10
                 if (j .eq. 11) ib = 1
                 if (k .ge. 1 .and. k .le. 10) then
                    call dchcel(l, ia, ib, ic, t)
                 end if
 1003         continue
 1004      continue
        end if

        if (ii .eq. 3) then
           j = i2 - 2
           if (j .le. 0) j = j + 10
           ib = j
           do 1006 i = i1 - 1, i1 + 1
              do 1005 k = i3 - 1, i3 + 1
                 ia = i
                 ic = k
                 if (i .eq. 0) ia = 10
                 if (i .eq. 11) ia = 1
                 if (k .ge. 1 .and. k .le. 10) then
                    call dchcel(l, ia, ib, ic, t)
                 end if
 1005         continue
 1006      continue
        end if

        if (ii .eq. 4) then
           j = i2 + 2
           if (j .ge. 11) j = j - 10
           ib = j
           do 1008 i = i1 - 1, i1 + 1
              do 1007 k = i3 - 1, i3 + 1
                 ia = i
                 ic = k
                 if (i .eq. 0) ia = 10
                 if (i .eq. 11) ia = 1
                 if (k .ge. 1 .and. k .le. 10) then
                    call dchcel(l, ia, ib, ic, t)
                 end if
 1007         continue
 1008      continue
        end if

        if (ii .eq. 5) then
           if (i3 .eq. 2) goto 100
           k = i3 - 2
           ic = k
           do 1010 i = i1 - 1, i1 + 1
              do 1009 j = i2 - 1, i2 + 1
                 ia = i
                 ib = j
                 if (i .eq. 0) ia = 10
                 if (i .eq. 11) ia = 1
                 if (j .eq. 0) ib = 10
                 if (j .eq. 11) ib = 1
                     call dchcel(l, ia, ib, ic, t)
 1009         continue
 1010      continue
        end if

        if (ii .eq. 6) then
           if (i3 .eq. 9) goto 100
           k = i3 + 2
           ic = k
           do 1012 i = i1 - 1, i1 + 1
              do 1011 j = i2 - 1, i2 + 1
                 ia = i
                 ib = j
                 if (i .eq. 0) ia = 10
                 if (i .eq. 11) ia = 1
                 if (j .eq. 0) ib = 10
                 if (j .eq. 11) ib = 1
                     call dchcel(l, ia, ib, ic, t)
 1011         continue
 1012      continue
        end if

 100        continue

        return
        end

        subroutine dchin3(l, ii, i1, i2, i3, t)
c       this subroutine is used to check collisions for particle inside
c       the cube when the collision just happened is a collision including 
c       collision with wall
c       and update the afftected particles through chkcel

c       input l,ii(specifying the direction of the wall collision),
c          i1,i2,i3, (specifying the position of the cell 
c                    we are going to check)
c          t

        implicit double precision (a-h, o-z)
        SAVE   

        if (ii .eq. 1) then
           i = i1 - 2
           if (i .le. 0) i = i + 10
           ia = i
           do 1002 j = i2 - 1, i2 + 1
              do 1001 k = i3 - 1, i3 + 1
                 ib = j
                 ic = k
                 if (j .eq. 0) ib = 10
                 if (j .eq. 11) ib = 1
                 if (k .eq. 0) ic = 10
                 if (k .eq. 11) ic = 1
                 call dchcel(l, ia, ib, ic, t)
 1001         continue
 1002      continue
        end if

        if (ii .eq. 2) then
           i = i1 + 2
           if (i .ge. 11) i = i - 10
           ia = i
           do 1004 j = i2 - 1, i2 + 1
              do 1003 k = i3 - 1, i3 + 1
                 ib = j
                 ic = k
                 if (j .eq. 0) ib = 10
                 if (j .eq. 11) ib = 1
                 if (k .eq. 0) ic = 10
                 if (k .eq. 11) ic = 1
                 call dchcel(l, ia, ib, ic, t)
 1003         continue
 1004      continue
        end if

        if (ii .eq. 3) then
           j = i2 - 2
           if (j .le. 0) j = j + 10
           ib = j
           do 1006 i = i1 - 1, i1 + 1
              do 1005 k = i3 - 1, i3 + 1
                 ia = i
                 ic = k
                 if (i .eq. 0) ia = 10
                 if (i .eq. 11) ia = 1
                 if (k .eq. 0) ic = 10
                 if (k .eq. 11) ic = 1
                 call dchcel(l, ia, ib, ic, t)
 1005         continue
 1006      continue
        end if

        if (ii .eq. 4) then
           j = i2 + 2
           if (j .ge. 11) j = j - 10
           ib = j
           do 1008 i = i1 - 1, i1 + 1
              do 1007 k = i3 - 1, i3 + 1
                 ia = i
                 ic = k
                 if (i .eq. 0) ia = 10
                 if (i .eq. 11) ia = 1
                 if (k .eq. 0) ic = 10
                 if (k .eq. 11) ic = 1
                 call dchcel(l, ia, ib, ic, t)
 1007         continue
 1008      continue
        end if

        if (ii .eq. 5) then
           k = i3 - 2
           if (k .le. 0) k = k + 10
           ic = k
           do 1010 i = i1 - 1, i1 + 1
              do 1009 j = i2 - 1, i2 + 1
                 ia = i
                 ib = j
                 if (i .eq. 0) ia = 10
                 if (i .eq. 11) ia = 1
                 if (j .eq. 0) ib = 10
                 if (j .eq. 11) ib = 1
                     call dchcel(l, ia, ib, ic, t)
 1009         continue
 1010      continue
        end if

        if (ii .eq. 6) then
           k = i3 + 2
           if (k .ge. 11) k = k - 10
           ic = k
           do 1012 i = i1 - 1, i1 + 1
              do 1011 j = i2 - 1, i2 + 1
                 ia = i
                 ib = j
                 if (i .eq. 0) ia = 10
                 if (i .eq. 11) ia = 1
                 if (j .eq. 0) ib = 10
                 if (j .eq. 11) ib = 1
                     call dchcel(l, ia, ib, ic, t)
 1011         continue
 1012      continue
        end if
c
        return
        end

        subroutine dchcel(l, i, j, k, t)
c       this subroutine is used to recalculate next collision time for 
c       particles in the cell i,j,k if the next collision partener is l

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist2/ icell, icel(10, 10, 10)
cc      SAVE /ilist2/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        if (i .eq. 11 .or. j .eq. 11 .or. k .eq. 11) then
           if ( .not. (i .eq. 11 .and. j .eq. 11 .and.
     &     k .eq. 11)) stop 'cerr'
           m = icell
        else
           m = icel(i, j, k)
        end if

        if (m .eq. 0) return
        if (next(m) .eq. l) then
           tm = tlarge
           last0 = 0
           call reor(t, tm, m, last0)
        end if
        n = nic(m)
        if (n .eq. 0) return
        do 10 while(n .ne. m)
           if (next(n) .eq. l) then
              tm = tlarge
              last0 = 0
              call reor(t, tm, n, last0)
           end if
           n = nic(n)
 10        continue

        return
        end

        subroutine fixtim(l, t, tmin1, tmin, nc)
c       this subroutine is used to compare the collision time with wall tmin1
c       and new collision time with particles for particle l
c       when used in ulist, input nc may be 0, which indicates no particle
c       collisions happen before wall collision, of course, then tmin=tmin1

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        t=t
        k = nc
        if (tmin .lt. tmin1) then
           ot(l) = tmin
           if (ct(l) .lt. tmin1) then
              icsta(l) = 0
           else
              icsta(l) = icsta(l) + 10
           end if
           next(l) = k
        else if (tmin .eq. tmin1) then
           ot(l) = tmin
           if (nc .eq. 0) then
              next(l) = 0
           else
              icsta(l) = icsta(l) + 10
              next(l) = k
           end if
        else
           ot(l) = tmin1
           next(l) = 0
        end if
        
        return
        end

        subroutine ud2(i, j, t, tmin, nc)
c       this subroutine is used to update next(i), ct(i), ot(i),
c        and get tmin, nc for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok

        call isco(i, j, allok, tm, t1, t2)

        if (allok) then
c       tm eq tmin, change nc to make sure fixtime get the collision with both 
c       wall and particle

             if (tm .lt. tmin) then
                tmin = tm
                ct(j) = t2
                nc = i
                if (iconfg .eq. 3 .or. iconfg .eq. 5) then
                   dgxa(j) = jxa * 10d0 * size1
                   dgya(j) = jya * 10d0 * size2
                   if (iconfg .eq. 5) then
                      dgza(j) = jza * 10d0 * size3
                   end if
                end if
             end if

             if (tm .le. ot(i)) then
                ct(i) = t1
                icels0 = icels(i)
                i1 = icels0 / 10000
                i2 = (icels0 - i1 * 10000) / 100
                i3 = icels0 - i1 * 10000 - i2 * 100
                call wallc(i, i1, i2, i3, t, tmin1)
                call fixtim(i, t, tmin1, tm, j)
                if (iconfg .eq. 3 .or. iconfg .eq. 5) then
                   dgxa(i) = - jxa * 10d0 * size1
                   dgya(i) = - jya * 10d0 * size2
                   if (iconfg .eq. 5) then
                      dgza(i) = - jza * 10d0 * size3
                   end if
                end if
             end if

             if (tm .gt. ot(i) .and. next(i) .eq. j) then
                ct(i) = t1
                call reor(t, tm, i, j)
             end if

           else if (next(i) .eq. j) then

             tm = tlarge
                
             call reor(t, tm, i, j)

          end if

        return
        end

        subroutine isco(i, j, allok, tm, t1, t2)

        implicit double precision (a-h, o-z)

        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        SAVE   

        logical allok

        iorder = iordsc / 10
        if (iconfg .eq. 1) then
           if (iorder .eq. 1) then
              call isco1(i, j, allok, tm, t1, t2)
           else if (iorder .eq. 2) then
              call isco2(i, j, allok, tm, t1, t2)
           else if (iorder .eq. 3) then
              call isco3(i, j, allok, tm, t1, t2)
           end if
        else if (iconfg .eq. 2 .or. iconfg .eq. 4) then
           if (iorder .eq. 1) then
              call isco4(i, j, allok, tm, t1, t2)
           else if (iorder .eq. 2) then
              call isco5(i, j, allok, tm, t1, t2)
           else if (iorder .eq. 3) then
              call isco6(i, j, allok, tm, t1, t2)
           end if
        else if (iconfg .eq. 3) then
           if (iorder .eq. 1) then
              call isco7(i, j, allok, tm, t1, t2)
           else if (iorder .eq. 2) then
              call isco8(i, j, allok, tm, t1, t2)
           else if (iorder .eq. 3) then
              call isco9(i, j, allok, tm, t1, t2)
           end if
        else if (iconfg .eq. 5) then
           if (iorder .eq. 1) then
              call isco10(i, j, allok, tm, t1, t2)
           else if (iorder .eq. 2) then
              call isco11(i, j, allok, tm, t1, t2)
           else if (iorder .eq. 3) then
              call isco12(i, j, allok, tm, t1, t2)
           end if
        end if

        return
        end

        subroutine isco1(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

c       set up numbers for later calculations
        i1 = i
        i2 = j

        p4 = ft(i2) - ft(i1)
        p1 = gx(i2) - gx(i1)
        p2 = gy(i2) - gy(i1)
        p3 = gz(i2) - gz(i1)

        q4 = e(i1)
        q1 = px(i1)
        q2 = py(i1)
        q3 = pz(i1)

        r4 = e(i2)
        r1 = px(i2)
        r2 = py(i2)
        r3 = pz(i2)

        a = p4 * q4 - p1 * q1 - p2 * q2 - p3 * q3
        b = p4 * r4 - p1 * r1 - p2 * r2 - p3 * r3
        c = q4 * q4 - q1 * q1 - q2 * q2 - q3 * q3
        d = r4 * r4 - r1 * r1 - r2 * r2 - r3 * r3
        ee = q4 * r4 - q1 * r1 - q2 * r2 - q3 * r3
        f = p4 * p4 - p1 * p1 - p2 * p2 - p3 * p3

c       make sure particle 2 formed early
        h = a + b
        if (h .gt. 0d0) then
           g = a
           a = -b
           b = -g

           g = c
           c = d
           d = g

           i1 = j
           i2 = i
        end if

c       check the approaching criteria
        if (allok) then

           vp = a * d - b * ee

           allok = allok .and. vp .lt. 0d0

        end if

c       check the closest approach distance criteria
         if (allok) then

           dm2 = - f - (a ** 2 * d + b ** 2 * c - 2d0 * a * b * ee) /
     &           (ee ** 2 - c * d)

           allok = allok .and. dm2 .lt. cutof2

        end if

c       check the time criteria
        if (allok) then

           tc1 = ft(i1) - e(i1) * (a * d - b * ee) / (ee ** 2 - c * d)
           tc2 = ft(i2) + e(i2) * (b * c - a * ee) / (ee ** 2 - c * d)
           tm = 0.5d0 * (tc1 + tc2)

           allok = allok .and. tm .gt. ft(i) .and. tm .gt. ft(j)

        end if

c        check rts cut
        if (allok) then

           rts2 = (q4 + r4) ** 2 - (q1 + r1) ** 2
     &          - (q2 + r2) ** 2 - (q3 + r3) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if
          
        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else if (h .gt. 0d0) then
           t1 = tm
           t2 = tm
        else
           t1 = tm
           t2 = tm
        end if

        return
        end

        subroutine isco2(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

c       set up numbers for later calculations
        i1 = i
        i2 = j

        p4 = ft(i2) - ft(i1)
        p1 = gx(i2) - gx(i1)
        p2 = gy(i2) - gy(i1)
        p3 = gz(i2) - gz(i1)

        q4 = e(i1)
        q1 = px(i1)
        q2 = py(i1)
        q3 = pz(i1)

        r4 = e(i2)
        r1 = px(i2)
        r2 = py(i2)
        r3 = pz(i2)

        a = p4 * q4 - p1 * q1 - p2 * q2 - p3 * q3
        b = p4 * r4 - p1 * r1 - p2 * r2 - p3 * r3
        c = q4 * q4 - q1 * q1 - q2 * q2 - q3 * q3
        d = r4 * r4 - r1 * r1 - r2 * r2 - r3 * r3
        ee = q4 * r4 - q1 * r1 - q2 * r2 - q3 * r3
        f = p4 * p4 - p1 * p1 - p2 * p2 - p3 * p3

c       make sure particle 2 formed early
        h = a + b
        if (h .gt. 0d0) then
           g = a
           a = -b
           b = -g

           g = c
           c = d
           d = g

           i1 = j
           i2 = i
        end if

c       check the approaching criteria
        if (allok) then

           vp = a * d - b * ee

           allok = allok .and. vp .lt. 0d0

        end if

c       check the closest approach distance criteria
         if (allok) then

           dm2 = - f - (a ** 2 * d + b ** 2 * c - 2d0 * a * b * ee) /
     &          (ee ** 2 - c * d)

           allok = allok .and. dm2 .lt. cutof2

        end if

c       check the time criteria
        if (allok) then

           tc1 = ft(i1) - e(i1) * (a * d - b * ee)/(ee ** 2 - c * d)
           tc2 = ft(i2) + e(i2) * (b * c - a * ee)/(ee ** 2 - c * d)
           if (iordsc .eq. 20) then
              tm = min(tc1, tc2)
           else if (iordsc .eq. 21) then
              tm = 0.5d0 * (tc1 + tc2)
           else
              tm = max(tc1, tc2)
           end if

           allok = allok .and. tm .gt. ft(i) .and. tm .gt. ft(j)

        end if

c        check rts cut
        if (allok) then

           rts2 = (q4 + r4) ** 2 - (q1 + r1) ** 2
     &          - (q2 + r2) ** 2 - (q3 + r3) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if
          
        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else if (h .gt. 0d0) then
           t1 = tc2
           t2 = tc1
        else
           t1 = tc1
           t2 = tc2
        end if

        return
        end

        subroutine isco3(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/  
        SAVE   

        logical allok

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

        if (ft(i) .ge. ft(j)) then
           i1 = j
           i2 = i
        else 
           i1 = i
           i2 = j
        end if
        
        if (allok) then

           t1 = ft(i1)
           vx1 = vx(i1)
           vy1 = vy(i1)
           vz1 = vz(i1)

           t2 = ft(i2)

           dvx = vx(i2) - vx1
           dvy = vy(i2) - vy1
           dvz = vz(i2) - vz1

           dt = t2 - t1

           dx = gx(i2) - gx(i1) - vx1 * dt
           dy = gy(i2) - gy(i1) - vy1 * dt
           dz = gz(i2) - gz(i1) - vz1 * dt

           vp = dvx * dx + dvy * dy + dvz * dz

           allok = allok .and. vp .lt. 0d0

        end if

        if (allok) then

           v2= dvx * dvx + dvy * dvy + dvz * dvz

           if (v2 .eq. 0d0) then
              tm = tlarge
           else
              tm = t2 - vp / v2
           end if

c       note now tm is the absolute time

           allok = allok .and. tm .gt. t1 .and. tm .gt. t2

        end if

        if (allok) then

           dgx = dx - dvx * t2
           dgy = dy - dvy * t2
           dgz = dz - dvz * t2

           dm2 = - v2 * tm ** 2  + dgx * dgx + dgy * dgy + dgz * dgz

           allok = allok .and. dm2 .lt. cutof2

        end if
        
        if (allok) then

           e1 = e(i1)
           px1 = px(i1)
           py1 = py(i1)
           pz1 = pz(i1)
           e2 = e(i2)
           px2 = px(i2)
           py2 = py(i2)
           pz2 = pz(i2)

           rts2 = (e1 + e2) ** 2 - (px1 + px2) ** 2
     &          - (py1 + py2) ** 2 - (pz1 + pz2) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if

        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else
           t1 = tm
           t2 = tm
        end if

        return
        end

        subroutine isco4(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

c       set up numbers for later calculations

        icels1 = icels(i)
        ii1 = icels1 / 10000
        jj1 = (icels1 - ii1 * 10000) / 100
        kk1 = icels1 - ii1 * 10000 - jj1 * 100
        icels2 = icels(j)
        ii2 = icels2 / 10000
        jj2 = (icels2 - ii2 * 10000) / 100
        kk2 = icels2 - ii2 * 10000 - jj2 * 100

        i1 = i
        i2 = j

        p4 = ft(i2) - ft(i1)
        p1 = gx(i2) - gx(i1)
        p2 = gy(i2) - gy(i1)
        p3 = gz(i2) - gz(i1)

        if (ii1 - ii2 .gt. 5) then
           p1 = p1 + 10d0 * size1
        else if (ii1 - ii2 .lt. -5) then
           p1 = p1 - 10d0 * size1
        end if
        if (jj1 - jj2 .gt. 5) then
           p2 = p2 + 10d0 * size2
        else if (jj1 - jj2 .lt. -5) then
           p2 = p2 - 10d0 * size2
        end if
        if (kk1 - kk2 .gt. 5) then
           p3 = p3 + 10d0 * size3
        else if (kk1 - kk2 .lt. -5) then
           p3 = p3 - 10d0 * size3
        end if

        q4 = e(i1)
        q1 = px(i1)
        q2 = py(i1)
        q3 = pz(i1)

        r4 = e(i2)
        r1 = px(i2)
        r2 = py(i2)
        r3 = pz(i2)

        a = p4 * q4 - p1 * q1 - p2 * q2 - p3 * q3
        b = p4 * r4 - p1 * r1 - p2 * r2 - p3 * r3
        c = q4 * q4 - q1 * q1 - q2 * q2 - q3 * q3
        d = r4 * r4 - r1 * r1 - r2 * r2 - r3 * r3
        ee = q4 * r4 - q1 * r1 - q2 * r2 - q3 * r3
        f = p4 * p4 - p1 * p1 - p2 * p2 - p3 * p3

c       make sure particle 2 formed early
        h = a + b
        if (h .gt. 0d0) then
           g = a
           a = -b
           b = -g

           g = c
           c = d
           d = g

           i1 = j
           i2 = i
        end if

c       check the approaching criteria
        if (allok) then

           vp = a * d - b * ee

           allok = allok .and. vp .lt. 0d0

        end if

c       check the closest approach distance criteria
         if (allok) then

           dm2 = - f - (a ** 2 * d + b ** 2 * c - 2d0 * a * b * ee) /
     &           (ee ** 2 - c * d)

           allok = allok .and. dm2 .lt. cutof2

        end if

c       check the time criteria
        if (allok) then

           tc1 = ft(i1) - e(i1) * (a * d - b * ee) / (ee ** 2 - c * d)
           tc2 = ft(i2) + e(i2) * (b * c - a * ee) / (ee ** 2 - c * d)
           tm = 0.5d0 * (tc1 + tc2)

           allok = allok .and. tm .gt. ft(i) .and. tm .gt. ft(j)

        end if

c        check rts cut
        if (allok) then

           rts2 = (q4 + r4) ** 2 - (q1 + r1) ** 2
     &          - (q2 + r2) ** 2 - (q3 + r3) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if
          
        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else if (h .gt. 0d0) then
           t1 = tm
           t2 = tm
        else
           t1 = tm
           t2 = tm
        end if

        return
        end

        subroutine isco5(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

c       set up numbers for later calculations

        icels1 = icels(i)
        ii1 = icels1 / 10000
        jj1 = (icels1 - ii1 * 10000) / 100
        kk1 = icels1 - ii1 * 10000 - jj1 * 100
        icels2 = icels(j)
        ii2 = icels2 / 10000
        jj2 = (icels2 - ii2 * 10000) / 100
        kk2 = icels2 - ii2 * 10000 - jj2 * 100

        i1 = i
        i2 = j

        p4 = ft(i2) - ft(i1)
        p1 = gx(i2) - gx(i1)
        p2 = gy(i2) - gy(i1)
        p3 = gz(i2) - gz(i1)

        if (ii1 - ii2 .gt. 5) then
           p1 = p1 + 10d0 * size1
        else if (ii1 - ii2 .lt. -5) then
           p1 = p1 - 10d0 * size1
        end if
        if (jj1 - jj2 .gt. 5) then
           p2 = p2 + 10d0 * size2
        else if (jj1 - jj2 .lt. -5) then
           p2 = p2 - 10d0 * size2
        end if
        if (kk1 - kk2 .gt. 5) then
           p3 = p3 + 10d0 * size3
        else if (kk1 - kk2 .lt. -5) then
           p3 = p3 - 10d0 * size3
        end if

        q4 = e(i1)
        q1 = px(i1)
        q2 = py(i1)
        q3 = pz(i1)

        r4 = e(i2)
        r1 = px(i2)
        r2 = py(i2)
        r3 = pz(i2)

        a = p4 * q4 - p1 * q1 - p2 * q2 - p3 * q3
        b = p4 * r4 - p1 * r1 - p2 * r2 - p3 * r3
        c = q4 * q4 - q1 * q1 - q2 * q2 - q3 * q3
        d = r4 * r4 - r1 * r1 - r2 * r2 - r3 * r3
        ee = q4 * r4 - q1 * r1 - q2 * r2 - q3 * r3
        f = p4 * p4 - p1 * p1 - p2 * p2 - p3 * p3

c       make sure particle 2 formed early
        h = a + b
        if (h .gt. 0d0) then
           g = a
           a = -b
           b = -g

           g = c
           c = d
           d = g

           i1 = j
           i2 = i
        end if

c       check the approaching criteria
        if (allok) then

           vp = a * d - b * ee

           allok = allok .and. vp .lt. 0d0

        end if

c       check the closest approach distance criteria
         if (allok) then

           dm2 = - f - (a ** 2 * d + b ** 2 * c - 2d0 * a * b * ee) /
     &           (ee ** 2 - c * d)

           allok = allok .and. dm2 .lt. cutof2

        end if

c       check the time criteria
        if (allok) then

           tc1 = ft(i1) - e(i1) * (a * d - b * ee) / (ee ** 2 - c * d)
           tc2 = ft(i2) + e(i2) * (b * c - a * ee) / (ee ** 2 - c * d)
           if (iordsc .eq. 20) then
              tm = min(tc1, tc2)
           else if (iordsc .eq. 21) then
              tm = 0.5d0 * (tc1 + tc2)
           else
              tm = max(tc1, tc2)
           end if

           allok = allok .and. tm .gt. ft(i) .and. tm .gt. ft(j)

        end if

c        check rts cut
        if (allok) then

           rts2 = (q4 + r4) ** 2 - (q1 + r1) ** 2
     &          - (q2 + r2) ** 2 - (q3 + r3) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if
          
        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else if (h .gt. 0d0) then
           t1 = tc2
           t2 = tc1
        else
           t1 = tc1
           t2 = tc2
        end if

        return
        end

        subroutine isco6(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

        if (ft(i) .ge. ft(j)) then
           i1 = j
           i2 = i
        else 
           i1 = i
           i2 = j
        end if

        icels1 = icels(i1)
        ii1 = icels1 / 10000
        jj1 = (icels1 - ii1 * 10000) / 100
        kk1 = icels1 - ii1 * 10000 - jj1 * 100
        icels2 = icels(i2)
        ii2 = icels2 / 10000
        jj2 = (icels2 - ii2 * 10000) / 100
        kk2 = icels2 - ii2 * 10000 - jj2 * 100
        
        if (allok) then

           t1 = ft(i1)
           vx1 = vx(i1)
           vy1 = vy(i1)
           vz1 = vz(i1)

           t2 = ft(i2)

           dvx = vx(i2) - vx1
           dvy = vy(i2) - vy1
           dvz = vz(i2) - vz1

           dt = t2 - t1

           dx = gx(i2) - gx(i1) - vx1 * dt
           dy = gy(i2) - gy(i1) - vy1 * dt
           dz = gz(i2) - gz(i1) - vz1 * dt

           if (ii1 - ii2 .gt. 5) then
              dx = dx + 10d0 * size1
           else if (ii1 - ii2 .lt. - 5) then
              dx = dx - 10d0 * size1
           end if

           if (jj1 - jj2 .gt. 5) then
              dy = dy + 10d0 * size2
           else if (jj1 - jj2 .lt. - 5) then
              dy = dy - 10d0 * size2
           end if

           if (kk1 - kk2 .gt. 5) then
              dz = dz + 10d0 * size3
           else if (kk1 - kk2 .lt. -5) then
              dz = dz - 10d0 * size3
           end if

           vp = dvx * dx + dvy * dy + dvz * dz

           allok = allok .and. vp .lt. 0d0

        end if

        if (allok) then

           v2p = dvx * dvx + dvy * dvy + dvz * dvz

           if (v2p .eq. 0d0) then
              tm = tlarge
           else
              tm = t2 - vp / v2p
           end if

c       note now tm is the absolute time

           allok = allok .and. tm .gt. t1 .and. tm .gt. t2

        end if

        if (allok) then

           dgx = dx - dvx * t2
           dgy = dy - dvy * t2
           dgz = dz - dvz * t2

           dm2 = - v2p * tm ** 2  + dgx * dgx + dgy * dgy + dgz * dgz

           allok = allok .and. dm2 .lt. cutof2

        end if
        
        if (allok) then

           e1 = e(i1)
           px1 = px(i1)
           py1 = py(i1)
           pz1 = pz(i1)
           e2 = e(i2)
           px2 = px(i2)
           py2 = py(i2)
           pz2 = pz(i2)

           rts2 = (e1 + e2) ** 2 - (px1 + px2) ** 2
     &          - (py1 + py2) ** 2 - (pz1 + pz2) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if

        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else
           t1 = tm
           t2 = tm
        end if

        return
        end

        subroutine isco7(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok, allokp

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

c       set up numbers for later calculations

        tm = tlarge

        if (allok) then
           do 1000 ii = - 1, 1
              do 2000 jj = - 1, 1

                 allokp = .true.
                 
                 i1 = i
                 i2 = j

                 p4 = ft(j) - ft(i)
                 p1 = gx(j) - gx(i)
                 p2 = gy(j) - gy(i)
                 p3 = gz(j) - gz(i)

                 p1 = p1 + ii * 10d0 * size1
                 p2 = p2 + jj * 10d0 * size2

                 q4 = e(i)
                 q1 = px(i)
                 q2 = py(i)
                 q3 = pz(i)
                 
                 r4 = e(j)
                 r1 = px(j)
                 r2 = py(j)
                 r3 = pz(j)

                 a = p4 * q4 - p1 * q1 - p2 * q2 - p3 * q3
                 b = p4 * r4 - p1 * r1 - p2 * r2 - p3 * r3
                 c = q4 * q4 - q1 * q1 - q2 * q2 - q3 * q3
                 d = r4 * r4 - r1 * r1 - r2 * r2 - r3 * r3
                 ee = q4 * r4 - q1 * r1 - q2 * r2 - q3 * r3
                 f = p4 * p4 - p1 * p1 - p2 * p2 - p3 * p3

c       make sure particle 2 formed early
                 h = a + b
                 if (h .gt. 0d0) then
                    g = a
                    a = -b
                    b = -g
                    g = c
                    c = d
                    d = g
                    i1 = j
                    i2 = i
                 end if
                 
c       check the approaching criteria
                 if (allokp) then
                    vp = a * d - b * ee
                    allokp = allokp .and. vp .lt. 0d0
                 end if

c       check the closest approach distance criteria
                 if (allokp) then
           dm2 = - f - (a ** 2 * d + b ** 2 * c - 2d0 * a * b * ee) /
     &            (ee ** 2 - c * d)
                    allokp = allokp .and. dm2 .lt. cutof2
                 end if

c       check the time criteria
                 if (allokp) then
           tc1 = ft(i1) - e(i1) * (a * d - b * ee) / (ee ** 2 - c * d)
           tc2 = ft(i2) + e(i2) * (b * c - a * ee) / (ee ** 2 - c * d)
           tmp = 0.5d0 * (tc1 + tc2)
           allokp = allokp .and. tmp .gt. ft(i) .and. tmp .gt. ft(j)
                 end if

                 if (allokp .and. tmp .lt. tm) then
                    tm = tmp
                    jxa = ii
                    jya = jj
cd                    dgxa(j) = ii * 10d0 * size1
cd                    dgya(j) = jj * 10d0 * size2
cd                    dgxa(i) = - dgxa(j)
cd                    dgya(i) = - dgya(j)
                 end if

 2000              continue
 1000           continue

           if (tm .eq. tlarge) then
              allok = .false.
           end if
           
        end if

c        check rts cut
        if (allok) then

           q4 = e(i1)
           q1 = px(i1)
           q2 = py(i1)
           q3 = pz(i1)

           r4 = e(i2)
           r1 = px(i2)
           r2 = py(i2)
           r3 = pz(i2)

           rts2 = (q4 + r4) ** 2 - (q1 + r1) ** 2
     &          - (q2 + r2) ** 2 - (q3 + r3) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if
          
        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else if (h .gt. 0d0) then
           t1 = tm
           t2 = tm
        else
           t1 = tm
           t2 = tm
        end if

        return
        end

        subroutine isco8(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok, allokp

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

c       set up numbers for later calculations

        tm = tlarge

        if (allok) then
           do 1000 ii = - 1, 1
              do 2000 jj = - 1, 1

                 allokp = .true.
                 
                 i1 = i
                 i2 = j

                 p4 = ft(j) - ft(i)
                 p1 = gx(j) - gx(i)
                 p2 = gy(j) - gy(i)
                 p3 = gz(j) - gz(i)

                 p1 = p1 + ii * 10d0 * size1
                 p2 = p2 + jj * 10d0 * size2

                 q4 = e(i)
                 q1 = px(i)
                 q2 = py(i)
                 q3 = pz(i)
                 
                 r4 = e(j)
                 r1 = px(j)
                 r2 = py(j)
                 r3 = pz(j)

                 a = p4 * q4 - p1 * q1 - p2 * q2 - p3 * q3
                 b = p4 * r4 - p1 * r1 - p2 * r2 - p3 * r3
                 c = q4 * q4 - q1 * q1 - q2 * q2 - q3 * q3
                 d = r4 * r4 - r1 * r1 - r2 * r2 - r3 * r3
                 ee = q4 * r4 - q1 * r1 - q2 * r2 - q3 * r3
                 f = p4 * p4 - p1 * p1 - p2 * p2 - p3 * p3

c       make sure particle 2 formed early
                 h = a + b
                 if (h .gt. 0d0) then
                    g = a
                    a = -b
                    b = -g
                    g = c
                    c = d
                    d = g
                    i1 = j
                    i2 = i
                 end if
                 
c       check the approaching criteria
                 if (allokp) then
                    vp = a * d - b * ee
                    allokp = allokp .and. vp .lt. 0d0
                 end if

c       check the closest approach distance criteria
                 if (allokp) then
           dm2 = - f - (a ** 2 * d + b ** 2 * c - 2d0 * a * b * ee) /
     &            (ee ** 2 - c * d)
                    allokp = allokp .and. dm2 .lt. cutof2
                 end if

c       check the time criteria
                 if (allokp) then
           tc1 = ft(i1) - e(i1) * (a * d - b * ee) / (ee ** 2 - c * d)
           tc2 = ft(i2) + e(i2) * (b * c - a * ee) / (ee ** 2 - c * d)
                    if (iordsc .eq. 20) then
                       tmp = min(tc1, tc2)
                    else if (iordsc .eq. 21) then
                       tmp = 0.5d0 * (tc1 + tc2)
                    else
                       tmp = max(tc1, tc2)
                    end if
           allokp = allokp .and. tmp .gt. ft(i) .and. tmp .gt. ft(j)
                 end if

                 if (allokp .and. tmp .lt. tm) then
                    tm = tmp
                    jxa = ii
                    jya = jj
                    ha = h
                    tc1a = tc1
                    tc2a = tc2
cd                    dgxa(j) = ii * 10d0 * size1
cd                    dgya(j) = jj * 10d0 * size2
cd                    dgxa(i) = - dgxa(j)
cd                    dgya(i) = - dgya(j)
                 end if

 2000              continue
 1000           continue

           if (tm .eq. tlarge) then
              allok = .false.
           end if
           
        end if

c        check rts cut
        if (allok) then

           q4 = e(i1)
           q1 = px(i1)
           q2 = py(i1)
           q3 = pz(i1)

           r4 = e(i2)
           r1 = px(i2)
           r2 = py(i2)
           r3 = pz(i2)

           rts2 = (q4 + r4) ** 2 - (q1 + r1) ** 2
     &          - (q2 + r2) ** 2 - (q3 + r3) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if
          
        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else if (ha .gt. 0d0) then
           t1 = tc2a
           t2 = tc1a
        else
           t1 = tc1a
           t2 = tc2a
        end if

        return
        end

        subroutine isco9(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok, allokp

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

        if (ft(i) .ge. ft(j)) then
           i1 = j
           i2 = i
           isign = -1
        else 
           i1 = i
           i2 = j
           isign = 1
        end if

        if (allok) then
           tm = tlarge
           
           t1 = ft(i1)
           vx1 = vx(i1)
           vy1 = vy(i1)
           vz1 = vz(i1)
           
           t2 = ft(i2)
           
           dvx = vx(i2) - vx1
           dvy = vy(i2) - vy1
           dvz = vz(i2) - vz1
           
           dt = t2 - t1

           do 1000 ii = - 1, 1
              do 2000 jj = - 1, 1

                 allokp = .true.

                 dx = gx(i2) - gx(i1) - vx1 * dt
                 dy = gy(i2) - gy(i1) - vy1 * dt
                 dz = gz(i2) - gz(i1) - vz1 * dt

                 dx = dx + ii * 10d0 * size1
                 dy = dy + jj * 10d0 * size2

                 vp = dvx * dx + dvy * dy + dvz * dz

                 allokp = allokp .and. vp .lt. 0d0
                 
                 if (allokp) then

                    v2 = dvx * dvx + dvy * dvy + dvz * dvz

                    if (v2 .eq. 0d0) then
                       tmp = tlarge
                    else
                       tmp = t2 - vp / v2
                    end if

c       note now tm is the absolute time

                    allokp = allokp .and. tmp .gt. t1 .and.
     &                         tmp .gt. t2

                 end if

                 if (allokp) then

                    dgx = dx - dvx * t2
                    dgy = dy - dvy * t2
                    dgz = dz - dvz * t2

                    dm2 = - v2 * tmp ** 2  + dgx * dgx +
     &                    dgy * dgy + dgz * dgz

                    allokp = allokp .and. dm2 .lt. cutof2

                 end if

                 if (allokp .and. tmp .lt. tm) then
                    tm = tmp
                    jxa = isign * ii
                    jya = isign * jj
                 end if

 2000              continue
 1000           continue
           
           if (tm .eq. tlarge) then
              allok = .false.
           end if
        end if
        
        if (allok) then

           e1 = e(i1)
           px1 = px(i1)
           py1 = py(i1)
           pz1 = pz(i1)
           e2 = e(i2)
           px2 = px(i2)
           py2 = py(i2)
           pz2 = pz(i2)

           rts2 = (e1 + e2) ** 2 - (px1 + px2) ** 2
     &          - (py1 + py2) ** 2 - (pz1 + pz2) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if

        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else
           t1 = tm
           t2 = tm
        end if

        return
        end

        subroutine isco10(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok, allokp

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

c       set up numbers for later calculations

        tm = tlarge

        if (allok) then
           do 1000 ii = - 1, 1
              do 2000 jj = - 1, 1
                 do 3000 kk = -1, 1
                 allokp = .true.
                 
                 i1 = i
                 i2 = j

                 p4 = ft(j) - ft(i)
                 p1 = gx(j) - gx(i)
                 p2 = gy(j) - gy(i)
                 p3 = gz(j) - gz(i)

                 p1 = p1 + ii * 10d0 * size1
                 p2 = p2 + jj * 10d0 * size2
                 p3 = p3 + kk * 10d0 * size3

                 q4 = e(i)
                 q1 = px(i)
                 q2 = py(i)
                 q3 = pz(i)
                 
                 r4 = e(j)
                 r1 = px(j)
                 r2 = py(j)
                 r3 = pz(j)

                 a = p4 * q4 - p1 * q1 - p2 * q2 - p3 * q3
                 b = p4 * r4 - p1 * r1 - p2 * r2 - p3 * r3
                 c = q4 * q4 - q1 * q1 - q2 * q2 - q3 * q3
                 d = r4 * r4 - r1 * r1 - r2 * r2 - r3 * r3
                 ee = q4 * r4 - q1 * r1 - q2 * r2 - q3 * r3
                 f = p4 * p4 - p1 * p1 - p2 * p2 - p3 * p3

c       make sure particle 2 formed early
                 h = a + b
                 if (h .gt. 0d0) then
                    g = a
                    a = -b
                    b = -g
                    g = c
                    c = d
                    d = g
                    i1 = j
                    i2 = i
                 end if
                 
c       check the approaching criteria
                 if (allokp) then
                    vp = a * d - b * ee
                    allokp = allokp .and. vp .lt. 0d0
                 end if

c       check the closest approach distance criteria
                 if (allokp) then
           dm2 = - f - (a ** 2 * d + b ** 2 * c - 2d0 * a * b * ee) /
     &            (ee ** 2 - c * d)
                    allokp = allokp .and. dm2 .lt. cutof2
                 end if

c       check the time criteria
                 if (allokp) then
           tc1 = ft(i1) - e(i1) * (a * d - b * ee) / (ee ** 2 - c * d)
           tc2 = ft(i2) + e(i2) * (b * c - a * ee) / (ee ** 2 - c * d)
           tmp = 0.5d0 * (tc1 + tc2)
           allokp = allokp .and. tmp .gt. ft(i) .and. tmp .gt. ft(j)
                 end if

                 if (allokp .and. tmp .lt. tm) then
                    tm = tmp
                    jxa = ii
                    jya = jj
                    jza = kk
cd                    dgxa(j) = ii * 10d0 * size1
cd                    dgya(j) = jj * 10d0 * size2
cd                    dgxa(i) = - dgxa(j)
cd                    dgya(i) = - dgya(j)
                 end if

 3000                 continue
 2000              continue
 1000           continue

           if (tm .eq. tlarge) then
              allok = .false.
           end if
           
        end if

c        check rts cut
        if (allok) then

           q4 = e(i1)
           q1 = px(i1)
           q2 = py(i1)
           q3 = pz(i1)

           r4 = e(i2)
           r1 = px(i2)
           r2 = py(i2)
           r3 = pz(i2)

           rts2 = (q4 + r4) ** 2 - (q1 + r1) ** 2
     &          - (q2 + r2) ** 2 - (q3 + r3) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if
          
        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else if (h .gt. 0d0) then
           t1 = tm
           t2 = tm
        else
           t1 = tm
           t2 = tm
        end if

        return
        end

        subroutine isco11(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok, allokp

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

c       set up numbers for later calculations

        tm = tlarge

        if (allok) then
           do 1000 ii = - 1, 1
              do 2000 jj = - 1, 1
                 do 3000 kk = - 1, 1

                 allokp = .true.
                 
                 i1 = i
                 i2 = j

                 p4 = ft(j) - ft(i)
                 p1 = gx(j) - gx(i)
                 p2 = gy(j) - gy(i)
                 p3 = gz(j) - gz(i)

                 p1 = p1 + ii * 10d0 * size1
                 p2 = p2 + jj * 10d0 * size2
                 p3 = p3 + kk * 10d0 * size3

                 q4 = e(i)
                 q1 = px(i)
                 q2 = py(i)
                 q3 = pz(i)
                 
                 r4 = e(j)
                 r1 = px(j)
                 r2 = py(j)
                 r3 = pz(j)

                 a = p4 * q4 - p1 * q1 - p2 * q2 - p3 * q3
                 b = p4 * r4 - p1 * r1 - p2 * r2 - p3 * r3
                 c = q4 * q4 - q1 * q1 - q2 * q2 - q3 * q3
                 d = r4 * r4 - r1 * r1 - r2 * r2 - r3 * r3
                 ee = q4 * r4 - q1 * r1 - q2 * r2 - q3 * r3
                 f = p4 * p4 - p1 * p1 - p2 * p2 - p3 * p3

c       make sure particle 2 formed early
                 h = a + b
                 if (h .gt. 0d0) then
                    g = a
                    a = -b
                    b = -g
                    g = c
                    c = d
                    d = g
                    i1 = j
                    i2 = i
                 end if
                 
c       check the approaching criteria
                 if (allokp) then
                    vp = a * d - b * ee
                    allokp = allokp .and. vp .lt. 0d0
                 end if

c       check the closest approach distance criteria
                 if (allokp) then
           dm2 = - f - (a ** 2 * d + b ** 2 * c - 2d0 * a * b * ee) /
     &            (ee ** 2 - c * d)
                    allokp = allokp .and. dm2 .lt. cutof2
                 end if

c       check the time criteria
                 if (allokp) then
           tc1 = ft(i1) - e(i1) * (a * d - b * ee) / (ee ** 2 - c * d)
           tc2 = ft(i2) + e(i2) * (b * c - a * ee) / (ee ** 2 - c * d)
                    if (iordsc .eq. 20) then
                       tmp = min(tc1, tc2)
                    else if (iordsc .eq. 21) then
                       tmp = 0.5d0 * (tc1 + tc2)
                    else
                       tmp = max(tc1, tc2)
                    end if
           allokp = allokp .and. tmp .gt. ft(i) .and. tmp .gt. ft(j)
                 end if

                 if (allokp .and. tmp .lt. tm) then
                    tm = tmp
                    jxa = ii
                    jya = jj
                    jza = kk
                    ha = h
                    tc1a = tc1
                    tc2a = tc2
cd                    dgxa(j) = ii * 10d0 * size1
cd                    dgya(j) = jj * 10d0 * size2
cd                    dgxa(i) = - dgxa(j)
cd                    dgya(i) = - dgya(j)
                 end if

 3000                 continue
 2000              continue
 1000           continue

           if (tm .eq. tlarge) then
              allok = .false.
           end if
           
        end if

c        check rts cut
        if (allok) then

           q4 = e(i1)
           q1 = px(i1)
           q2 = py(i1)
           q3 = pz(i1)

           r4 = e(i2)
           r1 = px(i2)
           r2 = py(i2)
           r3 = pz(i2)

           rts2 = (q4 + r4) ** 2 - (q1 + r1) ** 2
     &          - (q2 + r2) ** 2 - (q3 + r3) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if

        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else if (ha .gt. 0d0) then
           t1 = tc2a
           t2 = tc1a
        else
           t1 = tc1a
           t2 = tc2a
        end if

        return
        end

        subroutine isco12(i, j, allok, tm, t1, t2)
c       this subroutine is used to decide whether there is a collision between
c       particle i and j, if there is one allok=1, and tm gives the 
c       collision time, t1 the collision time for i,
c       t2 the collision time for j

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok, allokp

c       preventing consecutive collisions
        allok = last(i) .ne. j .or. last(j) .ne. i

        if (ft(i) .ge. ft(j)) then
           i1 = j
           i2 = i
           isign = -1
        else 
           i1 = i
           i2 = j
           isign = 1
        end if

        if (allok) then
           tm = tlarge
           
           t1 = ft(i1)
           vx1 = vx(i1)
           vy1 = vy(i1)
           vz1 = vz(i1)
           
           t2 = ft(i2)
           
           dvx = vx(i2) - vx1
           dvy = vy(i2) - vy1
           dvz = vz(i2) - vz1
           
           dt = t2 - t1

           do 1000 ii = - 1, 1
              do 2000 jj = - 1, 1
                 do 3000 kk = -1, 1

                 allokp = .true.

                 dx = gx(i2) - gx(i1) - vx1 * dt
                 dy = gy(i2) - gy(i1) - vy1 * dt
                 dz = gz(i2) - gz(i1) - vz1 * dt

                 dx = dx + ii * 10d0 * size1
                 dy = dy + jj * 10d0 * size2
                 dz = dz + kk * 10d0 * size3

                 vp = dvx * dx + dvy * dy + dvz * dz

                 allokp = allokp .and. vp .lt. 0d0
                 
                 if (allokp) then

                    v2 = dvx * dvx + dvy * dvy + dvz * dvz

                    if (v2 .eq. 0d0) then
                       tmp = tlarge
                    else
                       tmp = t2 - vp / v2
                    end if

c       note now tm is the absolute time

                    allokp = allokp .and. tmp .gt. t1 .and.
     &                         tmp .gt. t2

                 end if

                 if (allokp) then

                    dgx = dx - dvx * t2
                    dgy = dy - dvy * t2
                    dgz = dz - dvz * t2

                    dm2 = - v2 * tmp ** 2  + dgx * dgx +
     &                    dgy * dgy + dgz * dgz

                    allokp = allokp .and. dm2 .lt. cutof2

                 end if

                 if (allokp .and. tmp .lt. tm) then
                    tm = tmp
                    jxa = isign * ii
                    jya = isign * jj
                    jza = isign * kk
                 end if

 3000                 continue
 2000              continue
 1000           continue
           
           if (tm .eq. tlarge) then
              allok = .false.
           end if
        end if
        
        if (allok) then

           e1 = e(i1)
           px1 = px(i1)
           py1 = py(i1)
           pz1 = pz(i1)
           e2 = e(i2)
           px2 = px(i2)
           py2 = py(i2)
           pz2 = pz(i2)

           rts2 = (e1 + e2) ** 2 - (px1 + px2) ** 2
     &          - (py1 + py2) ** 2 - (pz1 + pz2) ** 2

           allok = allok .and. rts2 .gt. rscut2
        end if

        if (.not. allok) then
           tm = tlarge
           t1 = tlarge
           t2 = tlarge
        else
           t1 = tm
           t2 = tm
        end if

        return
        end

        subroutine reor(t, tmin, j, last0)
c       this subroutine is used to fix ct(i) when tm is greater than ct(i)
c       next(i) is last1 or last2

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
cd        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        icels0 = icels(j)

        i1 = icels0 / 10000
        i2 = (icels0 - i1 * 10000) / 100
        i3 = icels0 - i1 * 10000 - i2 * 100

        call wallc(j, i1, i2, i3, t, tmin1)

        if (tmin .le. tmin1) then
           nc = last0
        else
           tmin = tmin1
           nc = 0
        end if

        if (iconfg .eq. 3 .or. iconfg .eq. 5) then
           call chcell(j, i1, i2, i3, last0, t, tmin, nc)
        else
           if (i1 .eq. 11 .and. i2 .eq. 11 .and. i3 .eq. 11) then
              call chout(j, last0, t, tmin, nc)
           else
              if (iconfg .eq. 1) then
                 call chin1(j, i1, i2, i3, last0, t, tmin, nc)
              else if (iconfg .eq. 2) then
                 call chin2(j, i1, i2, i3, last0, t, tmin, nc)
              else if (iconfg .eq. 4) then
                 call chin3(j, i1, i2, i3, last0, t, tmin, nc)
              end if
           end if
        end if
        
        call fixtim(j, t, tmin1, tmin, nc)

        return
        end

        subroutine chout(l, last0, t, tmin, nc)
c       this subroutine is used to check the surface when the colliding
c       particle is outside the cube

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        SAVE   

        m1 = 11
        m2 = 11
        m3 = 11
        call chcell(l, m1, m2, m3, last0, t, tmin, nc)

        do 1003 i = 1, 10
           do 1002 j = 1, 10
              do 1001 k = 1, 10
                 if (i .eq. 1 .or. i .eq. 10 .or. j .eq. 1 .or.
     &              j .eq. 10 .or. k .eq. 1 .or. k. eq. 10)
     &               call chcell(l, i, j, k, last0, t, tmin, nc)
 1001         continue
 1002      continue
 1003   continue

        return
        end

        subroutine chin1(l, i1, i2, i3, last0, t, tmin, nc)
c       this subroutine is used to check collisions for particle inside
c       the cube
c       and update the afftected particles through chcell

        implicit double precision (a-h, o-z)
        SAVE   
        
c       itest is a flag to make sure the 111111 cell is checked only once
        itest = 0

        do 1003 i = i1 - 1, i1 + 1
           do 1002 j = i2 - 1, i2 + 1
              do 1001 k = i3 - 1, i3 + 1
                 if (i .ge. 1 .and. i .le. 10
     &              .and. j .ge. 1 .and. j .le. 10
     &              .and. k .ge. 1 .and. k .le. 10) then
                    call chcell(l, i, j, k, last0, t, tmin, nc)
                 else if (itest .eq. 0) then
                    m1 = 11
                    m2 = 11
                    m3 = 11
                    call chcell(l, m1, m2, m3, last0, t, tmin, nc)
                    itest = 1
                 end if   
 1001         continue
 1002      continue
 1003   continue

        return
        end

        subroutine chin2(l, i1, i2, i3, last0, t, tmin, nc)
c       this subroutine is used to check collisions for particle inside
c       the cube
c       and update the afftected particles through chcell

        implicit double precision (a-h, o-z)
        SAVE   
        
c       itest is a flag to make sure the 111111 cell is checked only once
        itest = 0

        do 1003 i = i1 - 1, i1 + 1
           do 1002 j = i2 - 1, i2 + 1
              do 1001 k = i3 - 1, i3 + 1
                 ia = i
                 ib = j
                 ic = k
                 if (k .ge. 1 .and. k .le. 10) then
                    if (i .eq. 0) ia = 10
                    if (i .eq. 11) ia = 1
                    if (j .eq. 0) ib = 10
                    if (j .eq. 11) ib = 1
                    call chcell(l, ia, ib, ic, last0, t, tmin, nc)
                 end if
 1001         continue
 1002      continue
 1003   continue

        return
        end

        subroutine chin3(l, i1, i2, i3, last0, t, tmin, nc)
c       this subroutine is used to check collisions for particle inside
c       the cube
c       and update the afftected particles through chcell

        implicit double precision (a-h, o-z)
        SAVE   
        
c       itest is a flag to make sure the 111111 cell is checked only once
        itest = 0

        do 1003 i = i1 - 1, i1 + 1
           do 1002 j = i2 - 1, i2 + 1
              do 1001 k = i3 - 1, i3 + 1
                 if (i .eq. 0) then
                    ia = 10
                 else if (i .eq. 11) then
                    ia = 1
                 else
                    ia = i
                 end if
                 if (j .eq. 0) then
                    ib = 10
                 else if (j .eq. 11) then
                    ib = 1
                 else
                    ib = j
                 end if
                 if (k .eq. 0) then
                    ic = 10
                 else if (k .eq. 11) then
                    ic = 1
                 else
                    ic = k
                 end if
                 call chcell(l, ia, ib, ic, last0, t, tmin, nc)
 1001         continue
 1002      continue
 1003   continue

        return
        end

        subroutine chcell(il, i1, i2, i3, last0, t, tmin, nc)
c       this program is used to check through all the particles, except last0
c       in the cell (i1,i2,i3) and see if we can get a particle collision 
c       with time less than the original input tmin ( the collision time of 
c       il with the wall
c       last0 cas be set to 0 if we don't want to exclude last0

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)
        
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist2/ icell, icel(10, 10, 10)
cc      SAVE /ilist2/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        SAVE   

        t=t
        if (iconfg .eq. 3 .or. iconfg .eq. 5) then
           jj = ichkpt
           do 1001 j = 1, jj
c     10/24/02 get rid of argument usage mismatch in mintm():
              jmintm=j
              if (j .ne. il .and. j .ne. last0)
     &          call mintm(il, jmintm, tmin, nc)
c     &          call mintm(il, j, tmin, nc)

 1001         continue
           return
        end if

c       set l
        if (i1 .eq. 11 .and. i2 .eq. 11 .and. i3 .eq. 11) then
           l = icell
        else
           l = icel(i1 ,i2, i3)
        end if

        if (l .eq. 0) return
        
        j = nic(l)
        
c       if there is only one particle
        if (j .eq. 0) then
           
c       if it's not il or last0,when last is not wall
           if (l .eq. il .or. l .eq. last0) return
           call mintm(il, l, tmin, nc)
           
c       if there are many particles
        else
           if (l .ne. il .and. l .ne. last0)
     &        call mintm(il, l, tmin, nc)
           do 10 while(j .ne. l)
              if (j .ne. il .and. j .ne. last0)
     &             call mintm(il, j, tmin, nc)
              j = nic(j)
 10           continue
        end if
        
        return
        end

        subroutine mintm(i, j, tmin, nc)
c       this subroutine is used to check whether particle j has smaller
c       collision time with particle i than other particles
c       or in other words, update next(i)

c       input i,j,tmin,nc
c       output tmin,nc

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /aurec1/ jxa, jya, jza
cc      SAVE /aurec1/
        common /aurec2/ dgxa(MAXPTN), dgya(MAXPTN), dgza(MAXPTN)
cc      SAVE /aurec2/
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        SAVE   

        logical allok

        call isco(i, j, allok, tm, t1, t2)

        if (allok .and. tm .lt. tmin) then
           tmin = tm
           ct(i) = t1
           nc = j
           if (iconfg .eq. 3 .or. iconfg .eq. 5) then
              dgxa(i) = - jxa * 10d0 * size1
              dgya(i) = - jya * 10d0 * size2
              if (iconfg .eq. 5) then
                 dgza(i) = - jza * 10d0 * size3
              end if
           end if
        end if

         return
        end

******************************************************************************
******************************************************************************

        subroutine zpca1

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        SAVE   

        if (mod(ictype,2) .eq. 0) then
           call zpca1a(iscat)
           call zpca1a(jscat)
clin-5/2009 ctest off v2 for parton:
c           call flowp(1)
        end if

        return
        end

        subroutine zpca1a(i)

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec3/gxs(MAXPTN),gys(MAXPTN),gzs(MAXPTN),fts(MAXPTN),
     &     pxs(MAXPTN), pys(MAXPTN), pzs(MAXPTN), es(MAXPTN),
     &     xmasss(MAXPTN), ityps(MAXPTN)
cc      SAVE /prec3/
        common /prec5/ eta(MAXPTN), rap(MAXPTN), tau(MAXPTN)
cc      SAVE /prec5/
        common /prec6/ etas(MAXPTN), raps(MAXPTN), taus(MAXPTN)
cc      SAVE /prec6/
        common /ana1/ ts(12)
cc      SAVE /ana1/
        SAVE   

        if (iconfg .eq. 1) then
           t1 = fts(i)
           t2 = ft(i)
           ipic = 11
        else if (iconfg .eq. 2 .or.
     &     iconfg .eq. 3) then
cd           t1 = fts(i)
cd           t2 = ft(i)
           t1 = taus(i)
           t2 = tau(i)
           ipic = 12
        else if (iconfg .eq. 4 .or.
     &     iconfg .eq. 5) then
           t1 = fts(i)
           t2 = ft(i)
           ipic = 12
        end if

        if (iconfg .le. 3) then
           do 1002 ian = 1, ipic
              if (t1 .le. ts(ian) .and.
     &           t2 .gt. ts(ian)) then
                 rapi = raps(i)
c     7/20/01:
c                 et = sqrt(pxs(i) ** 2 + pys(i) ** 2 + xmp ** 2)
                 et = dsqrt(pxs(i) ** 2 + pys(i) ** 2 + xmp ** 2)
                 call zpca1b(rapi, et, ian)
              end if
 1002      continue
        else
           do 1003 ian = 1, ipic
              if (t1 .le. ts(ian) .and.
     &           t2 .gt. ts(ian)) then
                 p0 = es(i)
                 p1 = pxs(i)
                 p2 = pys(i)
                 p3 = pzs(i)
                 call zpca1c(p0, p1, p2, p3, ian)
              end if
 1003      continue
        end if

        return
        end

        subroutine zpca1b(rapi, et, ian)

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para6/ centy
cc      SAVE /para6/
        common /ilist6/ t, iopern, icolln
cc      SAVE /ilist6/
        common /ana2/
     &     det(12), dn(12), detdy(12), detdn(12), dndy(12),
     &     det1(12), dn1(12), detdy1(12), detdn1(12), dndy1(12),
     &     det2(12), dn2(12), detdy2(12), detdn2(12), dndy2(12)
cc      SAVE /ana2/
        SAVE   

        if (rapi .gt. centy - 0.5d0 .and. 
     &     rapi .lt. centy + 0.5d0) then
           det2(ian) = det2(ian) + et
           dn2(ian) = dn2(ian) + 1d0
cdtrans
           if (ian .eq. 10) then
cd              write (10, *) t, det2(ian)
           end if
           if (ian .eq. 11) then
cd              write (11, *) t, det2(ian)
           end if
           if (ian .eq. 12) then
cd              write (12, *) t, det2(ian)
           end if
cdtransend
           if (rapi .gt. centy - 0.25d0 .and. 
     &        rapi .lt. centy + 0.25d0) then
              det1(ian) = det1(ian) + et
              dn1(ian) = dn1(ian) + 1d0
              if (rapi .gt. centy - 0.1d0 .and.
     &           rapi .lt. centy + 0.1d0) then
                 det(ian) = det(ian) + et
                 dn(ian) = dn(ian) + 1d0
              end if
           end if
        end if

        return
        end

        subroutine zpca1c(p0, p1, p2, p3, ian)

        implicit double precision (a-h, o-z)

        common /ana3/ em(4, 4, 12)
cc      SAVE /ana3/

        dimension en(4)
        SAVE   

        en(1) = p0
        en(2) = p1
        en(3) = p2
        en(4) = p3

        do 1002 i = 1, 4
           do 1001 j = 1, 4
              em(i, j, ian) = em(i, j, ian) + en(i) * en(j) / p0
 1001      continue
 1002   continue

        return
        end

******************************************************************************

        subroutine zpca2

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para3/ nsevt, nevnt, nsbrun, ievt, isbrun
cc      SAVE /para3/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /para7/ ioscar,nsmbbbar,nsmmeson
cc      SAVE /para7/
        common /ilist6/ t, iopern, icolln
cc      SAVE /ilist6/
        common /rndm1/ number
cc      SAVE /rndm1/
        common /rndm2/ iff
cc      SAVE /rndm2/
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        COMMON /AREVT/ IAEVT, IARUN, MISS
cc      SAVE /AREVT/
        SAVE   

        if (iconfg .le. 3) then
           call zpca2a
        else
           call zpca2b
        end if

        if (ioscar .eq. 1) then
           call zpca2c
        end if

cbzdbg2/17/99
c        write (25, *) 'Event', nsevt - 1 + ievt, 
c    &         ', run', isbrun,
c        WRITE (25, *) ' Event ', IAEVT, ', run ', IARUN,
c     &     ',\n\t number of operations = ', iopern,
c     &     ',\n\t number of collisions between particles = ', 
c     &         icolln,
c     &     ',\n\t freezeout time=', t,
c     &     ',\n\t ending at the ', number, 'th random number',
c     &     ',\n\t ending collision iff=', iff
cms     WRITE (25, *) ' Event ', IAEVT, ', run ', IARUN
cms     WRITE (25, *) '    number of operations = ', iopern
cms     WRITE (25, *) '    number of collisions between particles = ', 
cms  &       icolln
cms     WRITE (25, *) '    freezeout time=', t
cms     WRITE (25, *) '    ending at the ', number, 'th random number'
cms     WRITE (25, *) '    ending collision iff=', iff

        return
        end

        subroutine zpca2a

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /para1/ mul
cc      SAVE /para1/
        common /para2/ xmp, xmu, alpha, rscut2, cutof2
cc      SAVE /para2/
        common /para3/ nsevt, nevnt, nsbrun, ievt, isbrun
cc      SAVE /para3/
        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        common /para6/ centy
cc      SAVE /para6/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /prec5/ eta(MAXPTN), rap(MAXPTN), tau(MAXPTN)
cc      SAVE /prec5/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        common /ilist6/ t, iopern, icolln
cc      SAVE /ilist6/
        common /rndm1/ number
cc      SAVE /rndm1/
        common /rndm2/ iff
cc      SAVE /rndm2/
        common /rndm3/ iseedp
cc      SAVE /rndm3/
        common /ana1/ ts(12)
cc      SAVE /ana1/
        common /ana2/
     &     det(12), dn(12), detdy(12), detdn(12), dndy(12),
     &     det1(12), dn1(12), detdy1(12), detdn1(12), dndy1(12),
     &     det2(12), dn2(12), detdy2(12), detdn2(12), dndy2(12)
cc      SAVE /ana2/
        common /ana4/ fdetdy(24), fdndy(24), fdndpt(12)
cc      SAVE /ana4/

        logical iwrite
        data iwrite / .false. /
        SAVE   

        do 1004 i = 1, ichkpt
           rapi = rap(i)
c     7/20/01:
c           et = sqrt(px(i) ** 2 + py(i) ** 2 + xmp ** 2)
           et = dsqrt(px(i) ** 2 + py(i) ** 2 + xmp ** 2)

           do 1001 j = 1, 24
              if (rapi .gt. j + centy - 13d0 
     &           .and. rapi .lt. j  + centy - 12d0) then
                 fdetdy(j) = fdetdy(j) + et
                 fdndy(j) = fdndy(j) + 1d0
              end if
 1001      continue

           do 1002 j = 1, 12
              if (et .gt. 0.5d0 * (j - 1) .and.
     &           et .lt. 0.5d0 * j ) then
                 fdndpt(j) = fdndpt(j) + 1d0
              end if
 1002      continue

           if (iconfg .eq. 1) then
              t1 = ft(i)
              t2 = tlarge
              ipic = 11
           else
              t1 = tau(i)
              t2 = tlarge
              ipic = 12
           end if

           do 1003 ian = 1, ipic
              if (t1 .le. ts(ian) .and.
     &           t2 .gt. ts(ian)) then
                 call zpca1b(rapi, et, ian)
              end if
 1003      continue

           if (iconfg .eq. 1) then
              call zpca1b(rapi, et, 12)
           end if
 1004   continue

        do 1005 ian = 1, 12
          if ( iwrite ) then
           if (dn(ian) .eq. 0d0 .or. dn1(ian) .eq. 0d0 .or.
     &        dn2(ian) .eq. 0d0) then
              print *, 'event=', ievt
              print *, 'dn(', ian, ')=', dn(ian), 'dn1(', ian,
     &           ')=', dn1(ian), 'dn2(', ian, ')=', dn2(ian)
           end if
           endif
           detdy(ian) = detdy(ian) + det(ian)
           if (dn(ian) .ne. 0) then
              detdn(ian) = detdn(ian) + det(ian) / dn(ian)
           end if
           dndy(ian) = dndy(ian) + dn(ian)
           detdy1(ian) = detdy1(ian) + det1(ian)
           if (dn1(ian) .ne. 0) then
              detdn1(ian) = detdn1(ian) + det1(ian) / dn1(ian)
           end if
           dndy1(ian) = dndy1(ian) + dn1(ian)
           detdy2(ian) = detdy2(ian) + det2(ian)
           if (dn2(ian) .ne. 0) then
              detdn2(ian) = detdn2(ian) + det2(ian) / dn2(ian)
           end if
           dndy2(ian) = dndy2(ian) + dn2(ian)
 1005   continue

        return
        end

        subroutine zpca2b

        implicit double precision (a-h, o-z)
        parameter (MAXPTN=400001)

        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        common /ilist4/ ifmpt, ichkpt, indx(MAXPTN)
cc      SAVE /ilist4/
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        common /ana1/ ts(12)
cc      SAVE /ana1/
        SAVE   

        do 1002 i = 1, ichkpt
           t1 = ft(i)
           t2 = tlarge
           ipic = 12

           do 1001 ian = 1, ipic
              if (t1 .le. ts(ian) .and.
     &           t2 .gt. ts(ian)) then
                 p0 = e(i)
                 p1 = px(i)
                 p2 = py(i)
                 p3 = pz(i)
                 call zpca1c(p0, p1, p2, p3, ian)
              end if
 1001      continue
 1002   continue

        return
        end

        subroutine zpca2c

        implicit double precision (a-h, o-z)

        character*8 code, versn
        character*4 reffra
        integer aproj, zproj, atarg, ztarg, event
        parameter (MAXPTN=400001)

        common /para1/ mul
cc      SAVE /para1/
        common /prec2/gx(MAXPTN),gy(MAXPTN),gz(MAXPTN),ft(MAXPTN),
     &       px(MAXPTN), py(MAXPTN), pz(MAXPTN), e(MAXPTN),
     &       xmass(MAXPTN), ityp(MAXPTN)
cc      SAVE /prec2/
        SAVE   
        data nff/0/

c       file header
        if (nff .eq. 0) then
cms        write (26, 101) 'OSCAR1997A'
cms        write (26, 101) 'final_id_p_x'
           code = 'ZPC'
           versn = '1.0.1'
           aproj = -1
           zproj = -1
           atarg = -1
           ztarg = -1
           reffra = 'cm'
           ebeam = 0d0
           ntestp = 1
cms        write (26, 102) code, versn, aproj, zproj, atarg, ztarg,
cms  &        reffra, ebeam, ntestp
           nff = 1
           event = 1
           bimp = 0d0
           phi = 0d0
        end if

c       comment

c       event header
cms     write (26, 103) event, mul, bimp, phi

c       particles
        do 99 i = 1, mul
cms        write (26, 104) i, ityp(i),
cms  &        px(i), py(i), pz(i), e(i), xmass(i),
cms  &        gx(i), gy(i), gz(i), ft(i)
 99         continue

         event = event + 1

cyy 101        format (a12)
cyy 102        format (2(a8, 2x), '(',i3, ',',i6, ')+(',i3, ',', i6, ')',
cyy     &     2x, a4, 2x, e10.4, 2x, i8)
cyy 103        format (i10, 2x, i10, 2x, f8.3, 2x, f8.3)
cyy 104        format (i10, 2x, i10, 2x, 9(e12.6, 2x))

        return
        end

******************************************************************************

        subroutine zpcou

        implicit double precision (a-h, o-z)

        common /para5/ iconfg, iordsc
cc      SAVE /para5/
        SAVE   

        if (iconfg .le. 3) then
           call zpcou1
        else
           call zpcou2
        end if

        return
        end

        subroutine zpcou1

        implicit double precision (a-h, o-z)

        common /para3/ nsevt, nevnt, nsbrun, ievt, isbrun
cc      SAVE /para3/
        common /ana1/ ts(12)
cc      SAVE /ana1/
        common /ana2/
     &     det(12), dn(12), detdy(12), detdn(12), dndy(12),
     &     det1(12), dn1(12), detdy1(12), detdn1(12), dndy1(12),
     &     det2(12), dn2(12), detdy2(12), detdn2(12), dndy2(12)
cc      SAVE /ana2/
        common /ana4/ fdetdy(24), fdndy(24), fdndpt(12)
cc      SAVE /ana4/
        SAVE   
c
        dpt = 0.5d0
        dy2 = 1d0
        dy1 = 0.5d0
        dy = 0.2d0
        ntotal = nevnt * nsbrun
c
        return
        end

        subroutine zpcou2

        implicit double precision (a-h, o-z)

        common /para3/ nsevt, nevnt, nsbrun, ievt, isbrun
cc      SAVE /para3/
        common /ilist3/ size1, size2, size3, v1, v2, v3, size
cc      SAVE /ilist3/
        common /ana1/ ts(12)
cc      SAVE /ana1/
        common /ana3/ em(4, 4, 12)
cc      SAVE /ana3/
        SAVE   
c
cms     open (28, file = 'ana4/em.dat', status = 'unknown')
        vol = 1000.d0 * size1 * size2 * size3
        ntotal = nevnt * nsbrun

        do 1002 ian = 1, 12
cms        write (28, *) '*** for time ', ts(ian), 'fm(s)'
           do 1001 i = 1, 4
cms           write (28, *) em(i, 1, ian) / vol / ntotal,
cms  &                        em(i, 2, ian) / vol / ntotal,
cms  &                        em(i, 3, ian) / vol / ntotal,
cms  &                        em(i, 4, ian) / vol / ntotal
 1001      continue
 1002   continue

        return
        end

******************************************************************************

      subroutine lorenz(energy, px, py, pz, bex, bey, bez)

c     add in a cut for beta2 to prevent gam to be nan (infinity)

      implicit double precision (a-h, o-z)

      common /lor/ enenew, pxnew, pynew, pznew
cc      SAVE /lor/
      SAVE   

      beta2 = bex ** 2 + bey ** 2 + bez ** 2
      if (beta2 .eq. 0d0) then
         enenew = energy
         pxnew = px
         pynew = py
         pznew = pz
      else
         if (beta2 .gt. 0.999999999999999d0) then
            beta2 = 0.999999999999999d0
            print *,'beta2=0.999999999999999'
         end if
clin-7/20/01:
c         gam = 1.d0 / sqrt(1.d0 - beta2)
         gam = 1.d0 / dsqrt(1.d0 - beta2)
         enenew = gam * (energy - bex * px - bey * py - bez * pz)
         pxnew = - gam * bex * energy + (1.d0 
     &        + (gam - 1.d0) * bex ** 2 / beta2) * px
     &        + (gam - 1.d0) * bex * bey/beta2 * py
     &        + (gam - 1.d0) * bex * bez/beta2 * pz     
         pynew = - gam * bey * energy 
     &        + (gam - 1.d0) * bex * bey / beta2 * px
     &        + (1.d0 + (gam - 1.d0) * bey ** 2 / beta2) * py
     &        + (gam - 1.d0) * bey * bez / beta2 * pz         
         pznew = - gam * bez * energy
     &        +  (gam - 1.d0) * bex * bez / beta2 * px
     &        + (gam - 1.d0) * bey * bez / beta2 * py
     &        + (1.d0 + (gam - 1.d0) * bez ** 2 / beta2) * pz    
      endif

      return
      end

      subroutine index1(n, m, arrin, indx)
c     indexes the first m elements of ARRIN of length n, i.e., outputs INDX
c     such that ARRIN(INDEX(J)) is in ascending order for J=1,...,m

      implicit double precision (a-h, o-z)

      dimension arrin(n), indx(n)
      SAVE   
      do 1001 j = 1, m
         indx(j) = j
 1001   continue
      l = m / 2 + 1
      ir = m
 10   continue
      if (l .gt. 1) then
         l = l - 1
         indxt = indx(l)
         q = arrin(indxt)
      else
         indxt = indx(ir)
         q = arrin(indxt)
         indx(ir) = indx(1)
         ir = ir - 1
         if (ir .eq. 1) then
            indx(1) = indxt
            return
         end if
      end if
      i = l
      j = l + l
 20   if (j .le. ir) then
         if (j .lt. ir) then
            if (arrin(indx(j)) .lt. arrin(indx(j + 1))) j = j + 1
         end if
         if (q .lt. arrin(indx(j))) then
            indx(i) = indx(j)
            i = j
            j = j + j
         else
            j = ir + 1
         end if
      goto 20
      end if
      indx(i) = indxt
      goto 10

      end


        double precision function ftime1(iseed)

c       this program is used to generate formation time
c       the calling program needs a common /par1/
c       and declare external ftime1

clin-8/19/02
        implicit double precision (a-h, o-z)

cc        external ran1

        parameter (hbarc = 0.197327054d0)

        common /par1/ formt
cc      SAVE /par1/
        SAVE   

        aa = hbarc / formt

clin7/20/01:
c        ftime1 = aa * sqrt(1d0 / ran1(iseed) - 1d0)
        ftime1 = aa * dsqrt(1d0 / ran1(iseed) - 1d0)
        return
        end


      subroutine cropro(vx1, vy1, vz1, vx2, vy2, vz2)

c     this subroutine is used to calculate the cross product of 
c     (vx1,vy1,vz1) and (vx2,vy2,vz2) and get the result (vx3,vy3,vz3)
c     and put the vector into common /cprod/

      implicit double precision (a-h, o-z)

      common/cprod/ vx3, vy3, vz3
cc      SAVE /cprod/
      SAVE   

      vx3 = vy1 * vz2 - vz1 * vy2
      vy3 = vz1 * vx2 - vx1 * vz2
      vz3 = vx1 * vy2 - vy1 * vx2

      return
      end
      
      subroutine xnormv(vx, vy, vz)

c      this subroutine is used to get a normalized vector 

      implicit double precision (a-h, o-z)
      SAVE   

clin-7/20/01:
c      vv = sqrt(vx ** 2 + vy ** 2 + vz ** 2)
      vv = dsqrt(vx ** 2 + vy ** 2 + vz ** 2)
      vx = vx / vv
      vy = vy / vv
      vz = vz / vv

      return
      end

cbz1/29/99
c      subroutine rotate(xn1, xn2, xn3, theta, v1, v2, v3)
      subroutine zprota(xn1, xn2, xn3, theta, v1, v2, v3)
cbz1/29/99end

c     this subroutine is used to rotate the vector (v1,v2,v3) by an angle theta
c     around the unit vector (xn1, xn2, xn3)

      implicit double precision (a-h, o-z)
      SAVE   

      vx = v1
      vy = v2
      vz = v3
      c = cos(theta)
      omc = 1d0 - c
      s = sin(theta)
      a11 = xn1 ** 2 * omc + c
      a12 = xn1 * xn2 * omc - s * xn3
      a13 = xn1 * xn3 * omc + s * xn2
      a21 = xn1 * xn2 * omc + s * xn3
      a22 = xn2 **2 * omc + c
      a23 = xn2 * xn3 * omc - s * xn1
      a31 = xn1 * xn3 * omc - s * xn2
      a32 = xn3 * xn2 * omc + s * xn1
      a33 = xn3 ** 2 * omc + c
      v1 = vx * a11 + vy * a12 + vz * a13
      v2 = vx * a21 + vy * a22 + vz * a23
      v3 = vx * a31 + vy * a32 + vz * a33
      
      return
      end

c      double precision function ran1(idum)

c*     return a uniform random deviate between 0.0 and 1.0. set idum to 
c*     any negative value to initialize or reinitialize the sequence.

c      implicit double precision (a-h, o-z)

c      dimension r(97)

c      common /rndm1/ number
cc      SAVE /rndm1/
c      parameter (m1 = 259200, ia1 = 7141, ic1 = 54773, rm1 = 1d0 / m1)
c      parameter (m2 = 134456, ia2 = 8121, ic2 = 28411, rm2 = 1d0 / m2)
c      parameter (m3 = 243000, ia3 = 4561, ic3 = 51349)
clin-6/23/00 save ix1-3:
clin-10/30/02 r unsaved, causing wrong values for ran1 when compiled with f77:
cc      SAVE ix1,ix2,ix3,r
c      SAVE   
c      data iff/0/

c      if (idum .lt. 0 .or. iff .eq. 0) then
c         iff = 1
c         ix1 = mod(ic1 - idum, m1)
c         ix1 = mod(ia1 * ix1 + ic1, m1)
c         ix2 = mod(ix1, m2)
c         ix1 = mod(ia1 * ix1 + ic1, m1)
c         ix3 = mod(ix1, m3)
c         do 11 j = 1, 97
c            ix1 = mod(ia1 * ix1 + ic1, m1)
c            ix2 = mod(ia2 * ix2 + ic2, m2)
c            r(j) = (dble(ix1) + dble(ix2) * rm2) * rm1
c 11         continue
c         idum = 1
c      end if
c      ix1 = mod(ia1 * ix1 + ic1, m1)
c      ix2 = mod(ia2 * ix2 + ic2, m2)
c      ix3 = mod(ia3 * ix3 + ic3, m3)
clin-7/01/02       j = 1 + (97 * i x 3) / m3
c      j=1+(97*ix3)/m3
clin-4/2008:
c      if (j .gt. 97 .or. j .lt. 1) pause
c      if (j .gt. 97 .or. j .lt. 1) print *, 'In zpc ran1, j<1 or j>97',j
c      ran1 = r(j)
c      r(j) = (dble(ix1) + dble(ix2) * rm2) * rm1

clin-6/23/00 check random number generator:
c      number = number + 1
c      if(number.le.100000) write(99,*) 'number, ran1=', number,ran1

c      return
c      end
