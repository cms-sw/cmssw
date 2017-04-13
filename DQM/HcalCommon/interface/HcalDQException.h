#ifndef HCALDQEXCEPTION_H
#define HCALDQEXCEPTION_H

/*
 *	file:		HcalDQException.h
 *	Author:		Viktor Khristenko
 *	Description:
 *		Exception Class
 *
 *	TODO:
 */

#include <exception>

namespace hcaldqm
{
	class HcalDQMException : public std::exception
	{
		public:
			HcalDQMException(const char* msg)
				: _msg(msg)
			{}

			HcalDQMException(std::string const& msg)
				: _msg(msg)
			{}

			virtual ~HcalDQMException() throw()
			{}

			virtual const char* what() const throw()
			{
				retunr _msg.c_str();
			}

		protected:
			std::string _msg;
	};
}

#endif
