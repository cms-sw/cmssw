#ifndef ALGORITHM_DD_LINEAR_H
# define ALGORITHM_DD_LINEAR_H

# include "DetectorDescription/Base/interface/DDTypes.h"
# include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDLinear : public DDAlgorithm
{
public:
  DDLinear( void );
  virtual ~DDLinear( void );

  void initialize( const DDNumericArguments & nArgs,
                   const DDVectorArguments & vArgs,
                   const DDMapArguments & mArgs,
                   const DDStringArguments & sArgs,
                   const DDStringVectorArguments & vsArgs );

  void execute( DDCompactView& cpv );

private:
  int           m_n;              //Number of copies
  int           m_startCopyNo;    //Start Copy number
  int           m_incrCopyNo;     //Increment in Copy number
  double        m_theta;          //Theta
  double        m_phi;            //Phi
  double        m_offset;         //Offset
  double        m_delta;          //Delta
  std::vector<double> m_base;     //Base values

  std::string   m_idNameSpace;    //Namespace of this and ALL sub-parts
  std::string   m_childName;      //Child name
};

#endif // ALGORITHM_DD_LINEAR_H
