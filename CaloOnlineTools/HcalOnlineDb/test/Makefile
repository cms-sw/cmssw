NAME=xmlToolsRun
CHANNEL_QUALITY_NAME=channelQualityRun
all:
	if [ -h $(NAME) ]; then rm $(NAME); fi
	if [ -h $(CHANNEL_QUALITY_NAME) ]; then rm $(CHANNEL_QUALITY_NAME); fi
	scramv1 b -r
	ln -s $(CMSSW_BASE)/test/$(SCRAM_ARCH)/$(NAME) $(NAME)
	ln -s $(CMSSW_BASE)/test/$(SCRAM_ARCH)/$(CHANNEL_QUALITY_NAME) $(CHANNEL_QUALITY_NAME)

cq:
	if [ -h $(CHANNEL_QUALITY_NAME) ]; then rm $(CHANNEL_QUALITY_NAME); fi
	scramv1 b -r
	ln -s $(CMSSW_BASE)/test/$(SCRAM_ARCH)/$(CHANNEL_QUALITY_NAME) $(CHANNEL_QUALITY_NAME)

purge:
	rm ../*/*~
