class FileExportPlugin(object):
  option_types={} #option: (name, type, default, extra...)
  plugin_name=''
  file_types=()
  def __init__(self):
    self.options={}
    for k,v in self.option_types.items():
      self.options[k]=v[2]
    
  def pluginName(self):
    return self.plugin_name
    
  def fileTypes(self):
    return self.file_types
    
  def listOptions(self):
    return self.option_types
    
  def setOption(self,option,value):
    check = self.checkOption(option,value)
    if check==True:
      self.options[option]=value
    else:
      raise check
    
  def getOption(self,option):
    return self.options.get(option,None)
    
  def checkOption(self,option,value):
    return True
  
  def export(self,data,filename,filetype):
    raise NotImplemented
