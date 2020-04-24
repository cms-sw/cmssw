from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

class Filter(Analyzer):
    
    def process(self, event):
        input_collection = getattr(event, self.cfg_ana.input_objects)
        output_collection = [obj for obj in input_collection \
                             if self.cfg_ana.filter_func(obj)]
        setattr(event, self.instance_label, output_collection)
