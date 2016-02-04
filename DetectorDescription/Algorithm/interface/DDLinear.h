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
  double        m_phi;            //Phi dir[Theta,Phi] ... unit-std::vector in direction Theta, Phi
  // double        m_offset;      //Offset - an offset distance in direction dir(Theta,Phi)
  // FIXME: Understand if the offset is needed.
  double        m_delta;          //Delta - distance between two subsequent positions along dir[Theta,Phi]
  std::vector<double> m_base;     //Base values - a 3d-point where the offset is calculated from
                                  //base is optional, if omitted base=(0,0,0)
  std::pair<std::string, std::string> m_childNmNs; //Child name
                                                   //Namespace of the child
};

#endif // ALGORITHM_DD_LINEAR_H
