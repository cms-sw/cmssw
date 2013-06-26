* User defined pomeron structure function routine. 
* B. Cox 15/05/2001
*
*  Input : X = x_{i/IP}
*          Q2 = photon virtuality
*
*  Output: XPQ(-6:6): PDG style array of partons, 0=gluon.
*          

      SUBROUTINE POMSTR(X,Q2,XPQ)
      DOUBLE PRECISION XPQ,X,Q2
      DIMENSION XPQ(-6:6)
      
CCC   Q2 avoid dummy argument warning. Only needed on gcc 4.3.4.
      Q2=Q2
      XPQ(1)=X*(1.0-X)
      XPQ(2)=X*(1.0-X)
*      XPQ(1)=X**-2
*      XPQ(2)=X**-2
      XPQ(3)=0.
      XPQ(4)=0.
      XPQ(5)=0.
      XPQ(6)=0.
      XPQ(-1)=X*(1.0-X)
      XPQ(-2)=X*(1.0-X)
*      XPQ(-1)=X**-2
*      XPQ(-2)=X**-2
      XPQ(-3)=0.
      XPQ(-4)=0.
      XPQ(-5)=0.
      XPQ(-6)=0.
      XPQ(0)=X*(1.0-X)

      RETURN
      END
