NAME=xmlToolsRun
all:
	if [ -h $(NAME) ]; then rm $(NAME); fi
	scramv1 b -r
	ln -s $(CMSSW_BASE)/test/$(SCRAM_ARCH)/$(NAME) $(NAME)

purge:
	rm ../*/*~
