      subroutine updatafile
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

c...data files.
c...record usual data: pt--bc p_t; rap--bc rapidity; shat--
c...the subprocess c.m.energy; y--rapidity but equal to 
c...ln((x1-x2)/(x1+x2)); pseta--bc pseudo-rapidity; 
c...z--(2(k1+k2).p_bc)/(square of the subprocess energy);

	open(unit=4,file='data/pt.dat',status='unknown')
	open(unit=5,file='data/rap.dat',status='unknown')
	open(unit=16,file='data/shat.dat',status='unknown')
	open(unit=7,file='data/y.dat',status='unknown')
	open(unit=8,file='data/pseta.dat',status='unknown')
      open(unit=9,file='data/z.dat',status='unknown')

c...record y-result with different pt cut: 5,...,100 gev.
      open(unit=21,file='data/pt005y.dat',status='unknown')
      open(unit=22,file='data/pt020y.dat',status='unknown')
      open(unit=23,file='data/pt035y.dat',status='unknown')
      open(unit=24,file='data/pt050y.dat',status='unknown')
      open(unit=25,file='data/pt100y.dat',status='unknown')

c...record p_t result with different y cut: |y|<0.5,...,2.5
	open(unit=31,file='data/y05pt.dat',status='unknown')
	open(unit=32,file='data/y10pt.dat',status='unknown')
	open(unit=33,file='data/y15pt.dat',status='unknown')
	open(unit=34,file='data/y20pt.dat',status='unknown')
	open(unit=35,file='data/y25pt.dat',status='unknown')

c...record shat-result with different y cut: |y|<0.5,...,2.5
	open(unit=41,file='data/y05shat.dat',status='unknown')
	open(unit=42,file='data/y10shat.dat',status='unknown')
	open(unit=43,file='data/y15shat.dat',status='unknown')
	open(unit=44,file='data/y20shat.dat',status='unknown')
	open(unit=45,file='data/y25shat.dat',status='unknown')

c...record shat-result with different p_t cut: p_t<5,...,100 gev
	open(unit=51,file='data/pt005shat.dat',status='unknown')
	open(unit=52,file='data/pt020shat.dat',status='unknown')
	open(unit=53,file='data/pt035shat.dat',status='unknown')
	open(unit=54,file='data/pt050shat.dat',status='unknown')
	open(unit=55,file='data/pt100shat.dat',status='unknown')

c...ya means |y|.
c...record p_t result with different y cut: 0.5<|y|<1.0,...
	open(unit=62,file='data/ya05-10pt.dat',status='unknown')
	open(unit=63,file='data/ya10-15pt.dat',status='unknown')
	open(unit=64,file='data/ya15-20pt.dat',status='unknown')
	open(unit=65,file='data/ya20-25pt.dat',status='unknown')

	end
