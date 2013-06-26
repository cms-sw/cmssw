import sys
import ConfigParser

class HDQMDatabaseProducerConfiguration:

    def __init__(self, configFileName):
        config = ConfigParser.ConfigParser()
        # config.read("HDQMDatabaseProducerConfiguration.cfg")
        config.read(configFileName)

        # [RunSelection]
        # ##############

        self.FirstRun = config.get('Config', 'FirstRun')
        self.LastRun = config.get('Config', 'LastRun')
        self.RunType = config.get('Config', 'RunType')

        if config.get('Config', 'DiscoverRecoTypes') == "True":
            self.RecoTypes = ""
        else:
            self.RecoTypes = config.get('Config', 'RecoTypes').split(",")

        self.RunsToSkip = config.get('Config', 'RunsToSkip')

        # Good run selection
        self.Group = config.get('Config', 'Group')

        # Tag
        # ###

        # Prefix to tag name. Will be composed with subdet and tag version to build the actual tag name
        self.TagName = config.get('Config', 'TagName')

        # Helper class to store information for each tag
        class SubDetInfo:
            def __init__(self, subDet, tag, qualityFlag):
                self.SubDet = subDet
                self.Tag = tag
                self.QualityFlag = qualityFlag

        self.SubDetsAndTags = list()
        fullList = config.get('Config', 'SubDetsAndTags')
        for item in fullList.split(";"):
            itemList = item.split(",")
            if len(itemList) != 3:
                print "Error: incorrect configuration of subDetsAndTags"
                sys.exit()
            self.SubDetsAndTags.append(SubDetInfo(itemList[0].strip(), itemList[1].strip(), itemList[2].strip()))
            # print itemList[0], itemList[1], itemList[2]

        # [Database]
        # ##########
        
        self.AuthenticationPath = config.get('Config', 'AuthenticationPath')
        self.Database = config.get('Config', 'Database')
        
        # [Directories]
        # #############
        
        # Directory where the scripts are
        self.BaseDir = config.get('Config', 'BaseDir')
        
        # CMS environment
        self.CMS_PATH = config.get('Config', 'CMS_PATH')
        self.CMSSW_Version = config.get('Config', 'CMSSW_Version')
        
        # Directory containing the cfg templates
        self.TemplatesDir = config.get('Config', 'TemplatesDir')
        
        # DQM root files location
        self.SourceDir = config.get('Config', 'SourceDir')
        
        # Directory where to copy the plots
        self.StorageDir = config.get('Config', 'StorageDir')
