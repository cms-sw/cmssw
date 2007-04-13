#ifndef PYQUEN_WRAPPER_H
#define PYQUEN_WRAPPER_H

/*
 *
 * Wrapper for FORTRAN version of PYQUEN
 *
 * Camelia Mironov
 *
 */

                                                                               
// PYQUEN routine declaration

#define PYQUEN pyquen_ 
    extern "C" {
      void PYQUEN(int& a,int& ifb,double& bfix);

    }



#endif  // HYQUEN_WRAPPER_H
