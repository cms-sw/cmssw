"""
This computes the most optimal COS_PHI_LUT and COSH_ETA_LUT. Call
:func:`~l1tGTSingleInOutLUT.SingleInOutLUT.export` to export the
generated LUT.
"""

import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTScales import scale_parameter
from statistics import mean, median, stdev
import math


class SingleInOutLUT:

    def __init__(self, width_in, unused_lsbs, lsb, output_scale_factor, operation, start_value=0, label=""):
        self.debug_txt = ""
        input_scale_factor = 2**unused_lsbs * lsb
        self.unused_lsbs = unused_lsbs
        self.lsb = lsb
        signed_output = min([operation(input_scale_factor * (i + 0.5) + start_value)
                            for i in range(2**width_in)]) < 0

        self.width_out = math.ceil(math.log2(output_scale_factor *
                                             max([abs(operation(input_scale_factor * (i + 0.5) + start_value)) for i in range(2**width_in - 1)] + 
                                             [abs(operation(input_scale_factor * (2**width_in - 1) + start_value))])))

        if signed_output:
            self.width_out += 1

        self.debug_info(
            "***************************** {} LUT {} *****************************".format(operation.__name__, label))
        self.debug_info("Depth: {} x {} (addr x data)".format(width_in, self.width_out))
        self.debug_info("Scale: {}".format(output_scale_factor))

        self.width_in = width_in
        self.output_scale_factor = output_scale_factor
        self.input_scale_factor = input_scale_factor
        self.operation = operation
        self.start_value = start_value
        self.lut = cms.vint32(
            * ([round(output_scale_factor * operation(input_scale_factor * (i + 0.5) + start_value)) for i in range(2**width_in - 1)]
               + [round(output_scale_factor * operation(input_scale_factor * (2 ** width_in - 1) + start_value))]))

        self.print_error()

    def debug_info(self, msg):
        self.debug_txt += msg + "\n"

    def config(self):
        return cms.PSet(
            output_scale_factor=cms.double(self.output_scale_factor),
            unused_lsbs=cms.uint32(self.unused_lsbs),
            lut=self.lut,
            max_error=cms.double(self.max_error)
        )

    def export(self, filename: str):
        print(self.debug_txt)
        with open(filename, "w") as file:
            for value in self.lut:
                file.write("{:X}".format(int(value) & ((1 << self.width_out) - 1)
                                         ).rjust(math.ceil(self.width_out/4), '0') + "\n")

    @ staticmethod
    def optimal_scale_factor(width_in, max_width_out, unused_lsbs, lsb, operation, start_value=0):
        input_scale_factor = 2**unused_lsbs * lsb
        scale_factor = (2**max_width_out - 1) / max([abs(operation(input_scale_factor * (i + 0.5) + start_value))
                                                     for i in range(2**width_in)])
        return scale_factor

    def print_error(self):
        errors = [abs(self.lut[int(i/(2**self.unused_lsbs))]/self.output_scale_factor -
                      self.operation(i * self.lsb + self.start_value)) for i in range(2**(self.width_in + self.unused_lsbs))]

        self.max_error = max(errors)

        self.debug_info("Error: {:.5f} +/- {:.5f}, max: {:.5f}, total: {:.5f}, median: {:.5f}".format(
            mean(errors), stdev(errors), self.max_error, sum(errors), median(errors)))

        # mass_errors = [errors[i]/(2*self.operation(i * self.lsb + self.start_value)) for i in range(2**(self.width_in + self.unused_lsbs)) ]
        # self.debug_info("inv mass error: {:.5f} +/- {:.5f}, max: {:.5f}, total: {:.5f}, median: {:.5f}".format(
        #       mean(mass_errors), stdev(mass_errors), max(mass_errors), sum(mass_errors), median(mass_errors)))


COS_PHI_IN_WIDTH = 10   # not using 2 lsb and 1 msb (cos(x + pi) = -cos(x), x in [0, pi))
COSH_ETA_IN_WIDTH = 11  # not using 2 lsb and 1 msb (splitted LUT)
ISOLATION_WIDTH = 11

# Since we calculate cosh(dEta) - cos(dPhi); both must be on the same scale the difference should fit into 17 bits for the DSP
optimal_scale_factor = math.floor(
    (2**17 - 1) / (math.cosh((2**(COSH_ETA_IN_WIDTH + 2) - 1)*scale_parameter.eta_lsb.value()) + 1))

COS_PHI_LUT = SingleInOutLUT(
    COS_PHI_IN_WIDTH, 2, scale_parameter.phi_lsb.value(), optimal_scale_factor, math.cos)

# eta in [0, 2pi)
COSH_ETA_LUT = SingleInOutLUT(
    COSH_ETA_IN_WIDTH, 2, scale_parameter.eta_lsb.value(), optimal_scale_factor, math.cosh, 0, "[0, 2pi)")

# eta in [2pi, 4pi)
COSH_ETA_LUT_2 = SingleInOutLUT(
    COSH_ETA_IN_WIDTH, 2, scale_parameter.eta_lsb.value(),
    SingleInOutLUT.optimal_scale_factor(
        COSH_ETA_IN_WIDTH, 17, 2, scale_parameter.eta_lsb.value(), math.cosh, 2**13 * scale_parameter.eta_lsb.value()),
    math.cosh, 2**13 * scale_parameter.eta_lsb.value(), "[2pi, 4pi)")
