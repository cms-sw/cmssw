// Author: Muhammad Arham

#ifndef DataFormats_HGCalReco_TracksterSoA_h
#define DataFormats_HGCalReco_TracksterSoA_h

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

constexpr unsigned int MAX_VERTICES = 1000;
using probability_array = Eigen::Array<float, 8, 1>;
using vertex_array = Eigen::Array<unsigned int, MAX_VERTICES, 1>;
using vertex_multiplicity_array = Eigen::Array<float, MAX_VERTICES, 1>;

GENERATE_SOA_LAYOUT(TracksterSoALayout,
    SOA_COLUMN(float, regressed_energy),
    SOA_COLUMN(float, raw_energy),
    SOA_COLUMN(float, boundTime),
    SOA_COLUMN(float, time),
    SOA_COLUMN(float, timeError),
    SOA_COLUMN(float, raw_pt),
    SOA_COLUMN(float, raw_em_pt),
    SOA_COLUMN(float, raw_em_energy),
    SOA_COLUMN(int, seedIndex),
    SOA_COLUMN(int, track_idx),
    SOA_EIGEN_COLUMN(Eigen::Vector3f, barycenter),
    SOA_EIGEN_COLUMN(Eigen::Matrix3f, eigenvectors),
    SOA_EIGEN_COLUMN(Eigen::Vector3f, eigenvalues),
    SOA_EIGEN_COLUMN(Eigen::Vector3f, sigmas),
    SOA_EIGEN_COLUMN(Eigen::Vector3f, sigmasPCA),
    SOA_EIGEN_COLUMN(probability_array, id_probabilties),
    SOA_EIGEN_COLUMN(vertex_array, vertices),
    SOA_EIGEN_COLUMN(vertex_multiplicity_array, vertex_multiplicity)
)

using TracksterSoA = TracksterSoALayout<>;
using TracksterSoAView = TracksterSoA::View;
using TracksterSoAConstView = TracksterSoA::ConstView;

#endif