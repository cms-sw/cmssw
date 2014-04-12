include Makefile.mk

CPPFLAGS=$(SRC_INCLUDES) $(BOOST_INCLUDES) $(CPPUNIT_INCLUDES)
all: lib__class__.so Demo

Demo: lib__class__.so src/main.o
	$(LINKER) $+ $(LDFLAGS) -o $@

fc_objects := $(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

lib__class__.so: $(fc_objects)
	$(LINKER) $+ -shared -o $@

clean:
	rm -f *.o */*.o *.so
	rm -f Demo
	rm -f $(fc_objects)
