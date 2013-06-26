NAME=xmlToolsRun
CHANNEL_QUALITY_NAME=channelQualityRun
O2O_TAG_LIST_FILE_NAME=o2o_tag_list.txt
O2O_SCRIPT=o2o.py
O2O_POPCON_DBWRITE_TEMPLATE=dbwrite_o2o_template.py

all: link
	if [ -h $(NAME) ]; then rm $(NAME); fi
	if [ -h $(CHANNEL_QUALITY_NAME) ]; then rm $(CHANNEL_QUALITY_NAME); fi
	scramv1 b -r
	ln -s $(CMSSW_BASE)/test/$(SCRAM_ARCH)/$(NAME) $(NAME)
	ln -s $(CMSSW_BASE)/test/$(SCRAM_ARCH)/$(CHANNEL_QUALITY_NAME) $(CHANNEL_QUALITY_NAME)

cq:
	if [ -h $(CHANNEL_QUALITY_NAME) ]; then rm $(CHANNEL_QUALITY_NAME); fi
	scramv1 b -r
	ln -s $(CMSSW_BASE)/test/$(SCRAM_ARCH)/$(CHANNEL_QUALITY_NAME) $(CHANNEL_QUALITY_NAME)

link:
	if [ -h $(O2O_TAG_LIST_FILE_NAME) ]; then rm $(O2O_TAG_LIST_FILE_NAME); fi
	if [ -h $(O2O_SCRIPT) ]; then rm $(O2O_SCRIPT); fi
	if [ -h $(O2O_POPCON_DBWRITE_TEMPLATE) ]; then rm $(O2O_POPCON_DBWRITE_TEMPLATE); fi
	ln -s $(CMSSW_BASE)/src/CaloOnlineTools/HcalOnlineDb/test/o2o/$(O2O_TAG_LIST_FILE_NAME) $(O2O_TAG_LIST_FILE_NAME)
	ln -s $(CMSSW_BASE)/src/CaloOnlineTools/HcalOnlineDb/python/$(O2O_SCRIPT) $(O2O_SCRIPT)
	ln -s $(CMSSW_BASE)/src/CaloOnlineTools/HcalOnlineDb/python/$(O2O_POPCON_DBWRITE_TEMPLATE) $(O2O_POPCON_DBWRITE_TEMPLATE)

purge:
	rm ../*/*~
