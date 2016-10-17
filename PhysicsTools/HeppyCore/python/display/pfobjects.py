from ROOT import TPolyLine, TGraph, TArc, TEllipse, kGray
import numpy as np
import operator
import math

class Blob(object):
    def __init__(self, cluster):
        self.cluster = cluster
        pos = cluster.position
        radius = cluster.size()
        thetaphiradius = cluster.angular_size()
        # print radius
        color = 1
        if cluster.particle:
            if cluster.particle.pdgid() == 22 or cluster.particle.pdgid() == 11:
                color = 2
            else:
                color = 4
        max_energy = cluster.__class__.max_energy
        self.contour_xy = TEllipse(pos.X(), pos.Y(), radius)
        self.contour_yz = TEllipse(pos.Z(), pos.Y(), radius)   
        self.contour_xz = TEllipse(pos.Z(), pos.X(), radius)
        self.contour_thetaphi = TEllipse(math.pi/2. - pos.Theta(), pos.Phi(),
                                         thetaphiradius)
        contours = [self.contour_xy, self.contour_yz, self.contour_xz,
                    self.contour_thetaphi]
        iradius = radius * cluster.energy / max_energy
        ithetaphiradius = thetaphiradius * cluster.energy / max_energy
        self.inner_xy = TEllipse(pos.X(), pos.Y(), iradius)
        self.inner_yz = TEllipse(pos.Z(), pos.Y(), iradius)   
        self.inner_xz = TEllipse(pos.Z(), pos.X(), iradius)
        self.inner_thetaphi = TEllipse(math.pi/2. - pos.Theta(), pos.Phi(),
                                       ithetaphiradius)
        inners = [self.inner_xy, self.inner_yz, self.inner_xz,
                  self.inner_thetaphi]
        for contour in contours:
            contour.SetLineColor(color)
            contour.SetFillStyle(0)
        for inner in inners: 
            inner.SetFillColor(color)
            inner.SetFillStyle(3002)
            
    def draw(self, projection, opt=''):
        if projection == 'xy':
            self.contour_xy.Draw(opt+"psame")
            self.inner_xy.Draw(opt+"psame")
        elif projection == 'yz':
            self.contour_yz.Draw(opt+"psame")
            self.inner_yz.Draw(opt+"psame")
        elif projection == 'xz':
            self.contour_xz.Draw(opt+"psame")            
            self.inner_xz.Draw(opt+"psame")
        elif projection == 'ECAL_thetaphi':
            if self.cluster.layer == 'ecal_in':
                self.contour_thetaphi.Draw(opt+"psame")            
                self.inner_thetaphi.Draw(opt+"psame")
        elif projection == 'HCAL_thetaphi':
            if self.cluster.layer == 'hcal_in':
                self.contour_thetaphi.Draw(opt+"psame")            
                self.inner_thetaphi.Draw(opt+"psame")            
        else:
            raise ValueError('implement drawing for projection ' + projection )
        

class GTrajectory(object):

    draw_smeared_clusters = True
    
    def __init__(self, description, linestyle=1, linecolor=1):
        self.desc = description
        npoints = len(self.desc.points)
        self.graph_xy = TGraph(npoints)
        self.graph_yz = TGraph(npoints)
        self.graph_xz = TGraph(npoints)
        self.graph_thetaphi = TGraph(npoints)
        self.graphs = [self.graph_xy, self.graph_yz, self.graph_xz, self.graph_thetaphi]
        def set_graph_style(graph):
            graph.SetMarkerStyle(2)
            graph.SetMarkerSize(0.7)
            graph.SetLineStyle(linestyle)
            graph.SetLineColor(linecolor)
        set_graph_style(self.graph_xy)
        set_graph_style(self.graph_yz)
        set_graph_style(self.graph_xz)
        set_graph_style(self.graph_thetaphi)
        for i, point in enumerate(self.desc.points.values()):
            self.graph_xy.SetPoint( i, point.X(), point.Y() )
            self.graph_yz.SetPoint(i, point.Z(), point.Y() )
            self.graph_xz.SetPoint(i, point.Z(), point.X() )
            tppoint = point
            if i == 0:
                tppoint = description.p4().Vect()
            self.graph_thetaphi.SetPoint(i, math.pi/2. - tppoint.Theta(), tppoint.Phi() )
        clusters = self.desc.clusters_smeared \
                   if self.__class__.draw_smeared_clusters \
                   else self.desc.clusters
        self.blobs = map(Blob, clusters.values())            

    def set_color(self, color):
        for graph in self.graphs:
            graph.SetMarkerColor(color)
        
    def draw(self, projection, opt=''):
        for blob in self.blobs: 
            blob.draw(projection, opt)
        if projection == 'xy':
            self.graph_xy.Draw(opt+"psame")
        elif projection == 'yz':
            self.graph_yz.Draw(opt+"psame")
        elif projection == 'xz':
            self.graph_xz.Draw(opt+"psame")
        elif 'thetaphi' in projection:
            self.graph_thetaphi.Draw(opt+"psame")            
        else:
            raise ValueError('implement drawing for projection ' + projection )
            
class GStraightTrajectory(GTrajectory):
    def __init__(self, description):
        super(GStraightTrajectory, self).__init__(description,
                                                  linestyle=2, linecolor=1)

    def draw(self, projection):
        super(GStraightTrajectory, self).draw(projection, 'l')
   

class GHelixTrajectory(GTrajectory):    
    def __init__(self, description):
        super(GHelixTrajectory, self).__init__(description)
        helix = description.path
        self.helix_xy = TArc(helix.center_xy.X(),
                             helix.center_xy.Y(),
                             helix.rho, helix.phi_min, helix.phi_max)
        self.helix_xy.SetFillStyle(0)
        #TODO this is patchy,need to access the last point, whatever its name
        max_time = helix.time_at_z(description.points.values()[-1].Z())
        npoints = 100
        self.graphline_xy = TGraph(npoints)
        self.graphline_yz = TGraph(npoints)
        self.graphline_xz = TGraph(npoints)
        self.graphline_thetaphi = TGraph(npoints)
        for i, time in enumerate(np.linspace(0, max_time, npoints)):
            point = helix.point_at_time(time)
            self.graphline_xy.SetPoint(i, point.X(), point.Y())
            self.graphline_yz.SetPoint(i, point.Z(), point.Y())
            self.graphline_xz.SetPoint(i, point.Z(), point.X())
            tppoint = point
            if i == 0:
                tppoint = description.p4().Vect()
            self.graphline_thetaphi.SetPoint(i, math.pi/2.-tppoint.Theta(), tppoint.Phi())
        if abs(self.desc.pdgid()) in [11,13]:
            def set_graph_style(graph):
                graph.SetLineWidth(3)
                graph.SetLineColor(5)
            set_graph_style(self.graphline_xy)
            set_graph_style(self.graphline_xz)
            set_graph_style(self.graphline_yz)
            set_graph_style(self.graphline_thetaphi)


    def draw(self, projection):
        if projection == 'xy':
            # self.helix_xy.Draw("onlysame")
            self.graphline_xy.Draw("lsame")
        elif projection == 'yz':
            self.graphline_yz.Draw("lsame")
        elif projection == 'xz':
            self.graphline_xz.Draw("lsame")
        elif 'thetaphi' in projection:
            self.graphline_thetaphi.Draw("lsame")            
        else:
            raise ValueError('implement drawing for projection ' + projection )
        super(GHelixTrajectory, self).draw(projection)
        

class GTrajectories(list):
    
    def __init__(self, particles):
        for ptc in particles:
            is_neutral = abs(ptc.q())<0.5
            TrajClass = GStraightTrajectory if is_neutral else GHelixTrajectory
            gtraj = TrajClass(ptc)
            self.append(gtraj)
            # display.register(gtraj,1)

    def draw(self, projection):
        for traj in self:
            traj.draw(projection)
