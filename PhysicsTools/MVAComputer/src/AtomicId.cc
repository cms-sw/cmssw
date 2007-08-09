#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <memory>
#include <set>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

namespace { // anonymous
	struct StringLess {
		bool operator()(const char *id1, const char *id2) const
		{ return std::strcmp(id1, id2) < 0; }
	};

	class IdCache {
	    public:
		IdCache();
		~IdCache();

		inline const char *findOrInsert(const char *string) throw();

	    private:
		std::multiset<const char *, StringLess>	idSet;
		static std::allocator<char>		stringAllocator;
	};
} // anonymous namespace

IdCache::IdCache()
{
}

IdCache::~IdCache()
{
	for(std::multiset<const char*, StringLess>::iterator iter =
	    idSet.begin(); iter != idSet.end(); iter++)
		stringAllocator.deallocate(const_cast<char*>(*iter),
		                           std::strlen(*iter));
}

const char *IdCache::findOrInsert(const char *string) throw()
{
	std::multiset<const char*, StringLess>::iterator pos =
						idSet.lower_bound(string);
	if (pos != idSet.end() && std::strcmp(*pos, string) == 0)
		return *pos;

	std::size_t size = std::strlen(string) + 1;
	char *unique = stringAllocator.allocate(size);
	std::memcpy(unique, string, size);

	idSet.insert(pos, unique);

	return unique;
}

namespace PhysicsTools {

static IdCache atomicIdCache;

const char *AtomicId::lookup(const char *string) throw()
{
	if (string)
		return atomicIdCache.findOrInsert(string);

	return 0;
}

} // namespace PhysicsTools
