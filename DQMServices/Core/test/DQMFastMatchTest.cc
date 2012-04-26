#include <iostream>

#include <boost/utility.hpp>
#include "DQMServices/Core/interface/DQMStore.h"

/*
 * Test case for the fastmatch implementation used in DQMStore class
 *
 */

int main(int argc, char** argv)
{
	std::vector<std::string> input =
	{ "MyText", "Text" };

	std::vector<std::pair<std::string, unsigned int> > test_patterns =
	{
		{ "*Text", BOOST_BINARY(11) },
		{ "MyText*", BOOST_BINARY(01) },
		{ "MyTe*xt", BOOST_BINARY(01) },

		{ "*Text*", BOOST_BINARY(11) },
		{ "*Te*xt", BOOST_BINARY(11) },
		{ "MyT*ex*t", BOOST_BINARY(01) },

		{ "*Tex?", BOOST_BINARY(11) },
		{ "[]Text*", BOOST_BINARY(10) },
	};

	for (auto const& pattern : test_patterns)
	{
		fastmatch fm(pattern.first);

		size_t loc = 0;
		for (auto const& i : input)
		{
			bool res = fm.match(i);

			bool formal_result = ((pattern.second & (1 << loc)) > 0);
			if (res != formal_result)
			{
				std::cout << "Error: pattern " << pattern.first
						<< " did not work correctly on input " << i
						<< std::endl;
				return 1;
			}

			loc++;
		}
	}

	// test was ok
	return 0;
}
