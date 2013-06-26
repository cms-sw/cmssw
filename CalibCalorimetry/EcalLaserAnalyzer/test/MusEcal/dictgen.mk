src/$(PKGNAME)_dict_src.cc: $(DICTH_src)
	@echo "Generating dictionary $@..."
	@$(ROOTCINT) -f $@ -c $(filter -I% -D%,$(CXXFLAGS)) $^
