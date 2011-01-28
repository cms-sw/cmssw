/*
 * StringTokenize.h
 *
 *  Created on: Mar 25, 2010
 *      Author: diguida
 */

#ifndef STRINGTOKENIZE_H_
#define STRINGTOKENIZE_H_

#include <vector>
#include <string>

void stringTokenize(const std::string& str,std::vector<std::string>& tokens, char delimiter = ' ');

#endif /* STRINGTOKENIZE_H_ */
