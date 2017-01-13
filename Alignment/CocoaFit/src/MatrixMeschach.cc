//   COCOA class implementation file
//Id:  MatrixMeschach.C
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce
#include <iomanip>
#include <cmath>  


#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaFit/interface/MatrixMeschach.h"


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MatrixMeschach::MatrixMeschach()
{
  //-  std::cout << "creating matrix0 " << std::endl;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MatrixMeschach::~MatrixMeschach()
{
  //-  std::cout << "deleting matrix " << std::endl;
  M_FREE(_Mat);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MatrixMeschach::MatrixMeschach( ALIint NoLin, ALIint NoCol )
{
  //-  std::cout << "creating matrix " << std::endl;
  _NoLines = NoLin; 
  _NoColumns = NoCol;
  _Mat = m_get( NoLin, NoCol );
  //_data = new ALIdouble[NoCol][NoLin];
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MatrixMeschach::MatrixMeschach( const MatrixMeschach& mat )
{
  //-  std::cout << "creating matrixc " << std::endl;
  _NoLines = mat._Mat->m; 
  _NoColumns = mat._Mat->n;
  _Mat = m_get( mat.NoLines(), mat.NoColumns() );
  // std::cout <<  "copy const" << mat._Mat << _Mat << NoLines() << NoColumns() << Mat()->m << Mat()->n <<std::endl;
  copy( mat );
}
 

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::copy( const MatrixMeschach& mat) 
{
  //  if( ALIUtils::debug >= 5) std::cout <<  "copy matrix" << mat._Mat << " " << _Mat << " L " << mat.NoLines() << " C " << mat.NoColumns() << " l " <<  mat.Mat()->m << " c " << mat.Mat()->n <<std::endl;

  for( ALIint lin=0; lin < _NoLines; lin++ ) {
    for( ALIint col=0;  col < _NoColumns; col++ ) {
      _Mat->me[lin][col] = mat.MatNonConst()->me[lin][col];
    } 
  }
  //   m_copy( mat._Mat, _Mat );
} 


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MatrixMeschach& MatrixMeschach::operator=( const MatrixMeschach& mat )
{
  if ( mat._Mat != _Mat ) {
      _NoLines = mat._Mat->m;
      _NoColumns = mat._Mat->n; 
      M_FREE( _Mat );
      _Mat = m_get( mat.NoLines(), mat.NoColumns() );
      copy( mat );
  }
   if(ALIUtils::debug >= 9) std::cout <<  "operator=" << mat._Mat << _Mat << NoLines() << NoColumns() << Mat()->m << Mat()->n <<std::endl;
  return *this;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::operator*=( const MatrixMeschach& mat )
{
  //  std::cout << " multiply matrices " << std::endl;

  if( _NoColumns != mat.NoLines() ){
    std::cerr << " !! Trying two multiply two matrices when the number of columns of first one is not equal to number of files of second one " << std::endl;
    std::cout << " multiplying matrix " << _NoLines  << " x " << _NoColumns << " and " << mat.NoLines() << " x " << mat.NoColumns() << " results in " << _NoLines << " x " << mat.NoColumns() << std::endl;
  }
  _NoColumns = mat.NoColumns();

  MAT* tempmat = m_get( _NoColumns, _NoLines );
  m_transp( _Mat, tempmat); 
  //  M_FREE( _Mat );
  _Mat = m_get( _NoLines, mat.NoColumns() );
  mtrm_mlt( tempmat, mat._Mat, _Mat);

  //  _NoColumns = mat.NoColumns(); 
  //  M_FREE(tempmat);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::operator+=( const MatrixMeschach& mat )
{
  if(_NoLines != mat._NoLines  || _NoColumns != mat._NoColumns ) {
    std::cerr << "!!!! cannot sum two matrices with different size" << std::endl
         << "MATRIX 1:" << _NoLines << " X " << _NoColumns << std::endl
         << "MATRIX 2:" << mat._NoLines << " X " << mat._NoColumns << std::endl;
  }
  MAT* tempmat = m_get( _NoColumns, _NoLines );
  m_copy( _Mat, tempmat); 
  M_FREE( _Mat );
  _Mat = m_get( _NoLines, _NoColumns );
  m_add( tempmat, mat._Mat, _Mat);
  M_FREE(tempmat);

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::operator*=( const ALIdouble num )
{
  for (ALIuint ii=0; ii<_Mat->m; ii++) {
      for (ALIuint jj=0; jj<_Mat->n; jj++) {
	  _Mat->me[ii][jj] *= num;
      }
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MatrixMeschach operator*( const MatrixMeschach& mat1, const MatrixMeschach& mat2 )
{
  MatrixMeschach mat1copy = mat1;
  if( mat1copy.NoColumns() != mat2.NoLines() ){
    std::cerr << " !! Trying two multiply two matrices when the number of columns of first one is not equal to number of files of second one " << std::endl;
    std::cout << " multiplying matrix " << mat1copy.NoLines() << " x " << mat1copy.NoColumns() << " and " << mat2.NoLines() << " x " << mat2.NoColumns() << " results in " << mat1copy.NoLines() << " x " << mat2.NoColumns() << std::endl;
  }
  mat1copy.setNoColumns( mat2.NoColumns() );

  MAT* tempmat = m_get( mat1copy.NoColumns(), mat1copy.NoLines() );
  m_transp( mat1copy.MatNonConst(), tempmat);
 
  //M_FREE( _Mat );
  mat1copy.setMat( m_get( mat1copy.NoLines(), mat2.NoColumns() ) );
  mtrm_mlt( tempmat, mat2.MatNonConst(), mat1copy.MatNonConst());

  free(tempmat);

  return mat1copy;


  MatrixMeschach* matout = new MatrixMeschach( mat1copy );

  /*  if(ALIUtils::debug >= 9) std::cout << "add" << mat1copy.NoLines() << mat1copy.NoColumns()
       << mat2.NoLines() << mat2.NoColumns()  
       << matout.NoLines() << matout.NoColumns() << std::endl;
  if(ALIUtils::debug >= 9) std::cout << "addM" << mat1copy.Mat()->m << mat1copy.Mat()->n
				     << mat2.Mat()->m << mat2.Mat()->n
       << matout.Mat()->m << matout.Mat()->n << std::endl;
  */
  //  return MatrixMeschach( matout );
  return ( *matout );

}
 

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MatrixMeschach operator+( const MatrixMeschach& mat1, const MatrixMeschach& mat2 )
{
  MatrixMeschach matout( mat1 );
  matout += mat2;
   if(ALIUtils::debug >= 9) std::cout << "add" << mat1.NoLines() << mat1.NoColumns()
       << mat2.NoLines() << mat2.NoColumns()  
       << matout.NoLines() << matout.NoColumns() << std::endl;
   if(ALIUtils::debug >= 9) std::cout << "addM" << mat1.Mat()->m << mat1.Mat()->n
       << mat2.Mat()->m << mat2.Mat()->n
       << matout.Mat()->m << matout.Mat()->n << std::endl;
  return MatrixMeschach( matout );

}
 

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MatrixMeschach operator-( const MatrixMeschach& mat1, const MatrixMeschach& mat2 )
{
  MatrixMeschach matout( mat1 );
  MatrixMeschach matout2( mat2 );
  matout += (-1 * matout2);
  return MatrixMeschach( matout );
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MatrixMeschach operator*( const ALIdouble doub, const MatrixMeschach& mat )
{
  MatrixMeschach matout( mat );
  matout *= doub;
  return matout;
}

 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MatrixMeschach operator*( const MatrixMeschach& mat, const ALIdouble doub )
{
  return doub*mat;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::transpose()
{
  //   if(ALIUtils::debug >= 9)
  /*  std::cout << "transposed"  <<_NoLines<<_NoColumns;
  MAT* tempmat = m_get(_NoColumns, _NoLines);
  m_transp( _Mat, tempmat );
  std::cout << "transposed"  <<_NoLines<<_NoColumns;
  M_FREE( _Mat );
  _Mat = m_get(_NoColumns, _NoLines);
  std::cout << "transposed"  <<_NoLines<<_NoColumns;
  m_copy( tempmat, _Mat );
  std::cout << "transposed"  <<_NoLines<<_NoColumns;
  int ntemp = _NoLines;
  _NoLines = _NoColumns;
  _NoColumns = ntemp;
  M_FREE(tempmat);
  */

  //-  std::cout << "transposed"  <<_NoLines<<_NoColumns;
  MAT* tempmat = m_get(_NoColumns, _NoLines);
  m_transp( _Mat, tempmat );
  //-  std::cout << "transposed"  <<_NoLines<<_NoColumns;
  M_FREE( _Mat );
  _Mat = m_get(_NoColumns, _NoLines);
  //- std::cout << "transposed"  <<_NoLines<<_NoColumns;
  for( ALIint lin=0; lin < _NoColumns; lin++ ) {
    for( ALIint col=0;  col < _NoLines; col++ ) {
      //-  std::cout << "setting mat "  << lin << " " << col << std::endl;
      _Mat->me[lin][col] = tempmat->me[lin][col];
    }
  }
  //  m_copy( tempmat, _Mat );
  //-  std::cout << "transposed"  <<_NoLines<<_NoColumns;
  int ntemp = _NoLines;
  _NoLines = _NoColumns;
  _NoColumns = ntemp;
  M_FREE(tempmat);
 
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::inverse()
{
  if(ALIUtils::debug >= 9) std::cout << "inverse" << _NoLines << "C" << _NoColumns << std::endl;
   MAT* tempmat = m_get(_NoLines, _NoColumns);
   ALIdouble factor = 1000.;
   (*this) *= 1./factor;
   m_inverse( _Mat, tempmat );
   m_copy( tempmat, _Mat );
   (*this) *= 1./factor;
  M_FREE(tempmat);
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::AddData( ALIuint lin, ALIuint col, ALIdouble data ) 
{
  if ( lin >= _Mat->m || col >= _Mat->n ) {
      std::cerr << "EXITING: matrix has only " << _NoLines << " lines and "
           << _NoColumns << " columns " << std::endl;
      std::cerr << "EXITING: matrix has only " << _Mat->m << " lines and "
           << _Mat->n << " columns " << std::endl;
      std::cerr << " You tried to add data in line " << lin << " and column "
           << col << std::endl;
      std::exception();
  }
  _Mat->me[lin][col] = data;

} 


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble MatrixMeschach::operator () (int i, int j) const 
{
  return _Mat->me[i][j];
}  


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::EliminateLines( ALIint lin_first, ALIint lin_last )
{
  //-  return;
  if ( lin_last < lin_first ) {
       std::cerr << "EXITING: cannot Eliminate Lines in matrix if first line is " << 
      lin_first << " and lastt line is " << lin_last << std::endl;
      //t          std::exception();
       return;
  }
  ALIint dif = (lin_last - lin_first) + 1; 
  ALIint newANolin = NoLines() - dif; 
  ALIint newANocol = NoColumns();
  MatrixMeschach newmat( newANolin, newANocol );
  ALIint iin = 0;
  for ( ALIint ii=0; ii<NoLines(); ii++) {
      if( ii < lin_first  || ii > lin_last ) { 
          for ( ALIint jj=0; jj<NoColumns(); jj++) {
              newmat.AddData(iin, jj, (*this)(ii,jj) );
	        if(ALIUtils::debug >= 9) std::cout << iin << jj << "nmat" << newmat.Mat()->me[iin][jj] << std::endl;
	  }
          iin++;
      }
  }
  M_FREE( _Mat );
  _Mat = m_get( newANolin, newANocol );
  copy( newmat );
  _NoLines = _Mat->m; 
  _NoColumns = _Mat->n;
  Dump( "newnewmat" );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::EliminateColumns( ALIint lin_first, ALIint lin_last )
{
  //-  return;
  if ( lin_last < lin_first ) {
      std::cerr << "EXITING: cannot Eliminate Lines in matrix if first line is " << 
      lin_first << " and lastt line is " << lin_last << std::endl;
      //t      std::exception();
      return;
  }
  ALIint dif = (lin_last - lin_first) + 1; 
  ALIint newANolin = NoLines();
  ALIint newANocol = NoColumns() - dif;
  MatrixMeschach newmat( newANolin, newANocol );
  ALIint jjn = 0;
  for ( ALIint jj=0; jj<NoColumns(); jj++) {
      if( jj < lin_first  || jj > lin_last ) { 
          for ( ALIint ii=0; ii<NoLines(); ii++) {
              newmat.AddData( ii, jjn, (*this)(ii,jj) );
               if(ALIUtils::debug >= 9) std::cout << ii << jjn << "nmat" << newmat.Mat()->me[ii][jjn] << std::endl;
	  }
          jjn++;
      }
  }
  M_FREE( _Mat );
  _Mat = m_get( newANolin, newANocol );
  copy( newmat );
  _NoLines = _Mat->m; 
  _NoColumns = _Mat->n;
  Dump( "newnewmat" );
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::Dump( const ALIstring& mtext )
{
  ostrDump( std::cout, mtext); 
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::ostrDump( std::ostream& fout, const ALIstring& mtext )
{
  fout << "DUMPM@@@@@    " << mtext << "    @@@@@" << std::endl;
  fout << "Matrix is (_Mat)" << _Mat->m << "x" << _Mat->n << std::endl;
  fout << "Matrix is " << _NoLines << "x" << _NoColumns << std::endl;
  for (ALIuint ii=0; ii<_Mat->m; ii++) {
    for (ALIuint jj=0; jj<_Mat->n; jj++) {
      fout << std::setw(8) << _Mat->me[ii][jj] << " ";
    }
    fout << std::endl;
  }
  //  m_output(_Mat);

}
 

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MatrixMeschach::SetCorrelation(ALIint i1, ALIint i2, ALIdouble corr)
{  
  AddData(i1,i2,corr * sqrt( (*this)(i1,i1)*(*this)(i2,i2) ) );
  AddData(i2,i1,corr * sqrt( (*this)(i1,i1)*(*this)(i2,i2) ) );
   if(ALIUtils::debug >= 9) std::cout << i1<< i2<< corr << "CORR" << (*this)(i1,i1) << " " << (*this)(i2,i2) << std::endl;
  //-  std::cout << (*this)(i1,i1)*(*this)(i2,i2)  << std::endl;
  //- std::cout << sqrt( (*this)(i1,i1)*(*this)(i2,i2) ) << std::endl;
   if(ALIUtils::debug >= 9) std::cout << corr * sqrt( (*this)(i1,i1)*(*this)(i2,i2) ) << std::endl;

}


MatrixMeschach* MatrixByMatrix( const MatrixMeschach& mat1, const MatrixMeschach& mat2 )
{
 MatrixMeschach* matout = new MatrixMeschach( mat1 );
  if( matout->NoColumns() != mat2.NoLines() ){
    std::cerr << " !! Trying two multiply two matrices when the number of columns of first one is not equal to number of files of second one " << std::endl;
    //    std::cout << " multiplying matrix " << matout->NoLines() << " x " << matout->NoColumns() << " and " << mat2.NoLines() << " x " << mat2.NoColumns() << " results in " << matout->NoLines() << " x " << mat2.NoColumns() << std::endl;
  }
  matout->setNoColumns( mat2.NoColumns() );

  MAT* tempmat = m_get( matout->NoColumns(), matout->NoLines() );
  m_transp( matout->MatNonConst(), tempmat); 
  //M_FREE( _Mat );
  matout->setMat( m_get( matout->NoLines(), mat2.NoColumns() ) );
  mtrm_mlt( tempmat, mat2.MatNonConst(), matout->MatNonConst());


  /*  if(ALIUtils::debug >= 9) std::cout << "add" << matout->NoLines() << matout->NoColumns()
       << mat2.NoLines() << mat2.NoColumns()  
       << matout->NoLines() << matout->NoColumns() << std::endl;
  if(ALIUtils::debug >= 9) std::cout << "addM" << matout->Mat()->m << matout->Mat()->n
				     << mat2.Mat()->m << mat2.Mat()->n
       << matout->Mat()->m << matout->Mat()->n << std::endl;
  */
  //  return MatrixMeschach( matout );
  return ( matout );

}
