#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <memory>
#include <set>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

namespace PhysicsTools {

struct StringLess {
	bool operator()(const char *id1, const char *id2) const
	{ return std::strcmp(id1, id2) < 0; }
};

static std::multiset<const char *, StringLess> idSet;
static std::allocator<char> stringAllocator;

const char *AtomicId::lookup(const char *string) throw()
{
	if (!string)
		return 0;

	std::set<const char *, StringLess>::iterator pos =
						idSet.lower_bound(string);
	if (pos != idSet.end() && std::strcmp(*pos, string) == 0)
		return *pos;

	std::size_t size = std::strlen(string) + 1;
	char *unique = stringAllocator.allocate(size);
	std::memcpy(unique, string, size);

	idSet.insert(pos, unique);

	return unique;
}

} // namespace PhysicsTools
