// MCDB API: macro.hpp
// LCG MCDB project, Monte Carlo Data Base
// http://mcdb.cern.ch
// 
// Sergey Belov <Sergey.Belov@cern.ch>, 2007
//

#ifndef MCDB_MACRO_HPP_
#define MCDB_MACRO_HPP_ 1


#define FUNC_GET(type,class,func) type class::func() { return func##_; }

#define FUNC_SET(type,class,func) type class::func(const type func) { func##_=func; return func##_; }

#define FUNC_SETGET(type,class,func) FUNC_GET(type,class,func)\
			             FUNC_SET(type,class,func)


#endif
